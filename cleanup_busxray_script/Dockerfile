FROM python:latest
LABEL Maintainer="busxray"
COPY ./ ./
RUN pip install -r requirements.txt

COPY H:/busxray/annotations_adjusted ./
# We need to define the command to launch when we are going to run the image.
# We use the keyword 'CMD' to do that.
# The following command will execute "python ./main.py".
CMD [ "python", "./split_train_test_val.py" ]