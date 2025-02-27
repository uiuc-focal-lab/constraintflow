FROM ubuntu:22.04
WORKDIR /constraintflow

# Install software
RUN apt-get update
RUN apt-get install -y python3-pip

# Install PIP packages
RUN pip install z3-solver
RUN pip install antlr4-python3-runtime==4.7.2
RUN pip install matplotlib
RUN pip install tabulate
RUN pip install call_function_with_timeout
RUN pip install seaborn

# Get files
ADD . /constraintflow/
