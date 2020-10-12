FROM library/python:3.7

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

COPY requirements.txt ./requirements.txt
RUN pip install -r requirements.txt
COPY solver.py .
COPY . .


RUN mkdir /results

CMD ["/bin/bash"]
