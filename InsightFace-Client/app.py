import argparse
import base64
import logging
import multiprocessing
import os
import time
from distutils import util
from functools import partial
from itertools import chain, islice, cycle
from pathlib import Path

import msgpack
import numpy as np
import requests
import ujson
import uuid
from tqdm import tqdm
from itertools import chain

session = requests.Session()
session.trust_env = False

logging.basicConfig(
    level='INFO',
    format='%(asctime)s %(levelname)s - %(message)s',
    datefmt='[%H:%M:%S]',
)

def list_files(path:str, allowed_ext:list) -> list:
    return [
        str(os.path.join(dp, f)) 
        for dp, dn, filenames in os.walk(path) 
        for f in filenames 
        if os.path.splitext(f)[1] in allowed_ext
    ]


def to_chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def file2base64(path):
    with open(path, mode='rb') as fl:
        encoded = base64.b64encode(fl.read()).decode('ascii')
        return encoded


def save_crop(data, name):
    img = base64.b64decode(data)
    with open(name, mode="wb") as fl:
        fl.write(img)
        fl.close()


def to_bool(input):
    try:
        return bool(util.strtobool(input))
    except:
        return False


class IFRClient:

    def __init__(self, host: str = 'http://localhost', port: int = '18081'):
        self.server = f'{host}:{port}'
        self.sess = requests.Session()

    def server_info(self, server: str = None, show=True):
        if server is None:
            server = self.server

        info_uri = f'{server}/info'
        info = self.sess.get(info_uri).json()

        if show:
            server_uri = self.server
            backend_name = info['models']['inference_backend']
            det_name = info['models']['det_name']
            rec_name = info['models']['rec_name']
            rec_batch_size = info['models']['rec_batch_size']
            det_batch_size = info['models']['det_batch_size']
            det_max_size = info['models']['max_size']

            print(f'Server: {server_uri}\n'
                  f'    Inference backend:      {backend_name}\n'
                  f'    Detection model:        {det_name}\n'
                  f'    Detection image size:   {det_max_size}\n'
                  f'    Detection batch size:   {det_batch_size}\n'
                  f'    Recognition model:      {rec_name}\n'
                  f'    Recognition batch size: {rec_batch_size}')

        return info

    def extract(self, data: list,
                mode: str = 'paths',
                server: str = None,
                threshold: float = 0.6,
                extract_embedding=True,
                return_face_data=False,
                return_landmarks=False,
                embed_only=False,
                limit_faces=0,
                use_msgpack=True):

        if server is None:
            server = self.server

        extract_uri = f'{server}/extract'

        if mode == 'data':
            images = dict(data=data)
        elif mode == 'paths':
            images = dict(urls=data)

        req = dict(images=images,
                   threshold=threshold,
                   extract_ga=False,
                   extract_embedding=extract_embedding,
                   return_face_data=return_face_data,
                   return_landmarks=return_landmarks,
                   embed_only=embed_only,  # If set to true API expects each image to be 112x112 face crop
                   limit_faces=limit_faces,  # Limit maximum number of processed faces, 0 = no limit
                   api_ver='2',
                   msgpack=use_msgpack,
                   )

        resp = self.sess.post(extract_uri, json=req, timeout=7200)
        if resp.headers['content-type'] == 'application/x-msgpack':
            content = msgpack.loads(resp.content)
        else:
            content = ujson.loads(resp.content)

        images = content.get('data')
        for im in images:
            status = im.get('status')
            if status != 'ok':
                print(content.get('traceback'))
                break
            faces = im.get('faces', [])
            for i, face in enumerate(faces):
                norm = face.get('norm', 0)
                prob = face.get('prob')
                size = face.get('size')
                facedata = face.get('facedata')
                if facedata:
                    if size > 20 and norm > 14:
                        save_crop(facedata, f'crops/{i}_{size}_{norm:2.0f}_{prob}.jpg')

        return (data, content)


if __name__ == "__main__":
    description = """
    This client requires an already running Triton server with ArcFace + models. 

    Three modes are supported:
    - embed_only    ►  Returns the ArcFace embeddings without performing face detection. Requires images to be 112x112.
    - detect_only   ►  Performs face detection only! (Not Implemented)
    - all           ►  Performs face detection and returns the ArcFace embeddings. Images can be of any size.
    
    By default, it will delete the contents of the output directory if it already exists!"""

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--port', default=18081, type=int, help='server port')
    parser.add_argument('--host', default='http://localhost', type=str, help='server hostname or IP with protocol')
    parser.add_argument('-t', '--threads', default=10, type=int, help='number of worker processes')
    parser.add_argument('-b', '--batch', default=1, type=int, help='number of images per server request')
    parser.add_argument('-d', '--dir', type=str, help='path to directory with images', default='/data')
    parser.add_argument("-m", "--mode", default="all", nargs='?', choices=['embed_only' ,'detect_only', 'all'], help="run in embed only or detect + embed mode")
    parser.add_argument('-o', '--output', type=str, help='directory where the templates are saved', default="/output")
    parser.add_argument('--exclude', action=argparse.BooleanOptionalAction, help="exclude images with no face detected")
    parser.add_argument('-e', '--extension', nargs='+', help="allowed image extensions", default=['.jpeg', '.jpg', '.bmp', '.png', '.webp', '.tiff'])
    
    args = parser.parse_args()

    # if Path(args.output).exists():
    #     shutil.rmtree(args.output)

    Path(args.output).mkdir(exist_ok=True)
    (Path(args.output) / "templates").mkdir(exist_ok=True)
    (Path(args.output) / "summary").mkdir(exist_ok=True)

    template_directory = Path(args.output) / "templates"
    summary_directory = Path(args.output) / "summary"
    client = IFRClient(host=args.host, port=args.port)

    if args.mode == "all":
        embed = "True"
        embed_only = "False"
        config_fd = "True"
        config_emb = "True"
    elif args.mode == "embed_only":
        embed = 'True'
        embed_only = "True"
        config_fd = "True"
        config_emb = "False"
    elif args.mode == "detect_only":
        embed = "False"
        embed_only = "False"
        config_fd = "True"
        config_emb = "False"
    
    print('---')
    client.server_info(show=True)
    print('Configs:')
    print(f"    Face Detection:     {config_fd}")
    print(f"    Face Embedding:     {config_emb}")
    print(f'    Request Batch Size: {args.batch}')
    print(f"    Number of Workers:  {args.threads}")
    print('---')

    mode = 'paths'
    files = list_files(args.dir, args.extension)
    total = len(files)
    print(f"Total files detected: {total}")

    im_batches = [
        list(chunk)
        for chunk in to_chunks(files, args.batch)
    ]

    print("Processing Images / Batches ...")
    t0 = time.time()
    index = 0

    def write_results(results):
        files, response = results
        files = np.array(files)
        features = np.zeros(shape=(len(files), 512), dtype=np.float32)

        index = 0
        for r in response["data"]:
            faces = r["faces"]
            if faces: features[index] = faces[-1]["vec"]
            index += 1

        del response
        
        identifier = uuid.uuid4()
        if args.exclude:
            where_zeros = np.all(features == 0, axis=1)
            features = features[~where_zeros]
            removed_files = files[where_zeros]
            files = files[~where_zeros]
            missing_templates_filename = f"missing_templates_{identifier}.txt"
            missing_templates_path = summary_directory / missing_templates_filename

            if len(removed_files) > 0:
                with open(missing_templates_path, "w") as f:
                    for file in removed_files:
                        f.write(Path(file).name + "\n")
            
            del removed_files

        files = [Path(f) for f in files]
        if len(files) > 0:
            npylist = open(summary_directory / f"templates_{identifier}.txt", "w")
            for index, file in enumerate(files):
                file_path = str(Path(template_directory / (file.stem + ".npy")))
                npylist.write(file.stem + ".npy" + "\n")
                np.save(file_path, np.asarray(features[index]))
            npylist.close()

    with multiprocessing.Pool(processes=args.threads) as pool:
        pbar = tqdm(total=len(im_batches))

        arguments = (
            {
                "data": batch,
                "extract_embedding": to_bool(embed),
                "embed_only": to_bool(embed_only),
                "mode": mode,
                "limit_faces": 0
            }
            for batch in im_batches
        )

        async_results = [
            pool.apply_async(client.extract, kwds=args, callback=write_results)
            for args in arguments
        ]
            
        for result in async_results:
            result.wait()
            pbar.update(1)

    t1 = time.time()
    took = t1 - t0
    speed = total / took
    print(f"Took: {took:.3f} s. ({speed:.3f} im/sec)")

    print("\nCleaning up...")
    missing_templates = [
        f for f in list_files(str(summary_directory), ['.txt'])
        if Path(f).name.startswith("missing_templates_")
    ]

    found_templates = [
        f for f in list_files(str(summary_directory), ['.txt'])
        if Path(f).name.startswith("templates_")
    ]

    with open(summary_directory / "missing_templates.txt", "w") as f:
        text = (open(str(f)).read() for f in missing_templates)
        f.writelines(text)
    
    with open(summary_directory / "templates.txt", "w") as f:
        text = (open(str(f)).read() for f in found_templates)
        f.writelines(text)

    for f in tqdm(chain(missing_templates, found_templates)):
        Path(f).unlink(missing_ok=True)
    
    print("Done!")
