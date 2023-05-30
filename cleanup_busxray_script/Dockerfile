FROM python:latest
LABEL Maintainer="busxray"
COPY ./ ./
RUN pip3 install -r requirements.txt

# We need to define the command to launch when we are going to run the image.
# We use the keyword 'CMD' to do that.
# The following command will execute "python ./main.py".
CMD [ "python", "split_train_test_val.py", "--host=0.0.0.0" ]