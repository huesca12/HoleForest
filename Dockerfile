FROM python:3

ADD main.py /

RUN pip install click
RUN pip install colorama
RUN pip install joblib
RUN pip install numpy
RUN pip install pandas
RUN pip install scikit-learn

CMD ["python", "./main.py"]
