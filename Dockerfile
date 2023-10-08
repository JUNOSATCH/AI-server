FROM public.ecr.aws/lambda/python:3.8

COPY requirements.txt ./
RUN pip install --upgrade pip &&\
 yum install -y gcc &&\
 yum install -y git &&\
 python3.8 -m pip install -r requirements.txt -t . &&\
 pip install 'git+https://github.com/SKTBrain/KoBERT.git#egg=kobert_tokenizer&subdirectory=kobert_hf'

COPY app.py ./
COPY model.pt ./

CMD ["app.lambda_handler"]
