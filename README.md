# Cerebras-GPT

Run Cerebras-GPT in container!!

[Reference Site](https://nowokay.hatenablog.com/entry/2023/03/31/110604)

## Requirements

- Mac OS or Linux
- Docker or Finch

## Run Cerebras-GPT

use [Docker](https://www.docker.com/)

```sh
git clone git@github.com:Ryusei-0407/run-cerebras-gpt.git
cd run-cerebras-gpt
docker build -t cerebras .
docker run -it cerebras:latest
python VTSTech-GPT.py -m 590m -p "雨が降るときは" -c
```

use [Finch](https://github.com/runfinch/finch)

```sh
git clone git@github.com:Ryusei-0407/run-cerebras-gpt.git
cd run-cerebras-gpt
finch build -t cerebras .
finch run -it cerebras:latest
python VTSTech-GPT.py -m 590m -p "雨が降るときは" -c
```
