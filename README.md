# Facial Feature Extractor

This repository contains a server and client to run extraction of facial templates 😀.

## 🖥️ Server 

Note: You must be part of the sudoers to run the code below without `sudo`. See here for [details](https://docs.docker.com/engine/install/linux-postinstall/).

To launch the backend, you need to use the `server.sh` script. 

Some options available are shown in the table below: 

| Option              | Description                                        | Default         | Required |
|---------------------|----------------------------------------------------|-----------------|----------|
| --start-port        | Starting port range for containers                 | 18081           | No       |
| --gpus-start        | GPU device number to start with                    | 0               | No       |
| --gpus              | Number of gpus, starting from gpus-start, to use   | 1               | No       |
| --workers           | Number of backend processes per container          | 1               | No       |
| --fp16              | Use half-point precision                           | True            | No       |
| --detection-model   | Model use for face detection                       | scrfd_10g_gnkps | No       |
| --recognition-model | Model use for facial feature extraction            | glintr100       | No       |
| --models-path       | Directory where pre-downloaded models can be found | $PWD/models     | No       |
| --data-path         | Path where the data is located                     | None            | Yes      |

As an example, for 1 container (1 GPU / 1 worker), with the data directory being `/path/to/sample/data`, you would do:
```bash
./server.sh --data-path /path/to/sample/data
```

The data path must be the same for the client and the server. 

## 💻 Client

Note: You must be part of the sudoers to run the code below without `sudo`. See here for [details](https://docs.docker.com/engine/install/linux-postinstall/).

Navigate to the `InsightFace-Client` directory first and then build the image.
```
cd InsightFace-Client
docker build -t xavier/arcface:v0.8.2.0 .
```

To start the client, you should use the docker run command. In the example below, we specify the data directory as /data and the output directory as /output. Within the output directory, you will find cropped faces (if requested with the --save-crops flag) in the /output/crops subdirectory, templates in the /output/templates subdirectory, and a summary of enrollments and failures in /output/summary. Please note that to ensure crops are saved, regardless of the flag, the mode should be set to either all or detect_only. The default mode is all.

```bash
docker run --rm \
-v /home/xavier/Documents/git/ArcFace/sample:/data \
-v /home/xavier/Documents/git/ArcFace/output:/output \
--net host \
xavier/arcface:v0.8.2.0 \
--exclude \
--save-crops \
--extension ".JPG"
```

Additional options can be appended to the command. Invoke the command with `-h` to see more information. 

```bash
docker run --rm \
-v /home/xavier/Documents/git/ArcFace/sample:/data \
-v /home/xavier/Documents/git/ArcFace/output:/output \
--net host \
xavier/arcface:v0.8.2.0 -h
```

Alternatively, see the table below:
| Option      | Description                                  | Default                                             | Required |
|-------------|----------------------------------------------|-----------------------------------------------------|----------|
| --host      | Server hostname or IP                        | localhost                                           | No       |
| --port      | Server port                                  | 18081                                               | No       |
| --threads   | Number of worker processes for data dispatch | 10                                                  | No       |
| --batch     | Number of images per server request          | 1                                                   | No       |
| --dir       | Path to directory with images                | /data                                               | No       |
| --output    | Directory where templates are saved          | /output                                             | No       |
| --exclude   | Exclude images with no face detected         | False                                               | No       |
| --save-crops   | Saves crops when face detected         | False                                               | No       |
| --extension | Allowed image extensions                     | '.jpeg'<br />'.jpg'<br />'.bmp'<br />'.png'<br />'.webp'<br />'.tiff' | No       |