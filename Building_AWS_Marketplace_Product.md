## Building AWS Marketplace Product based on Model served On Sagemaker
Publishing an ML product to AWS marketplace can be accomplished by few ways. One way to achieve this is to bring your own model (BYOM) deployed in AWS Sagemaker as an Endpoint. In order to this we should 
1- Containerize your code and upload it to ECR.
2- Create the model package in AWS Sagemaker
3- Prepare the AWS Marketplace listing and the sample notebook



### 1-Containerize your code and upload it to ECR.
- Even though there are other options, our approach is to build a self contained model in docker container that is designed to be hosted in AWS Sagemaker.
- The distinction that we are doing this for marketplace serving is important. For the marketplace it is important that the container has all it's dependecies packaged because it's not possible to import anything from outside in a AWS marketplace product. (E.g the model weights can not be in S3 it should be inside the container).
- Even though the documentation states that you may use existing accounts, it is important that as a seller you must be a  resident or citizen of eligible jurisdiction. What these jurisdictions are can be found the aws marketplace guide. It's important that you should go over the relevant sections in the guide. [AWS Marketplace Seller Guide](https://pages.awscloud.com/rs/112-TZM-766/images/aws-marketplace-ug.pdf)

- Another important point is whether the product is able to retrain a model or is it only suitable for inference. In our case we will only provide inference, a.i we only accept predefined input and output predictions based on that input.(E.g a scalar sentiment scoring between 0, 1; 1 indicating positive 0 indicating negative sentiment)

- The approach detailed here is inference-only approach. For this the first step to prepare the folder structure that is detailed in AWS.
Taken from [here](https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/scikit_bring_your_own.ipynb)    
### The parts of the sample container  
- In the container directory are all the components you need to package the sample algorithm for Amazon SageMager:  
  
        |-- Dockerfile
        |-- build_and_push.sh 
         -- sentiment_analyser(main_container_directory)
            |-- nginx.conf
            |-- predictor.py
            |-- serve
            |-- train
             -- wsgi.py
    Let's discuss each of these in turn:  
    **Dockerfile**: describes how to build your Docker container image. More details below.  
    **build_and_push.sh**: is a script that uses the Dockerfile to build your container images and then pushes it to ECR. We'll invoke the commands directly later in this notebook, but you can just copy and run the script for your own algorithms.  
    **sentiment_analyer** is the directory which contains the files that will be installed in the container. You may name it according to your use case.  


    The files that we'll put in the container are:  
    **nginx.conf**: is the configuration file for the nginx front-end. **Generally, you should be able to take this file as-is.**
    **predictor.py**: is the program that actually implements the Flask web server and the decision tree predictions for this app. You'll want to customize the actual prediction parts to your application. Since this algorithm is simple, we do all the processing here in this file, but you may choose to have separate files for implementing your custom logic.  
    **serve**: is the program started when the container is started for hosting. It simply launches the gunicorn server which runs multiple instances of the Flask app defined in 'predictor.py'. **You should be able to take this file as-is.**
    **wsgi.py**: is a small wrapper used to invoke the Flask app. **You should be able to take this file as-is.**

    In summary, the one file you will probably want to change for your application is 'predictor.py'.  
    In our usecase I used the predictor.py file as a wrapper where I import the main python files that do the main heavy lifting.
```
import sentiment_analyzer 
```


the main parts of the predictor.py file are as follows:

we are creating an instance of SentimentAnalyer class, a specialized Turkish sentiment_analyser build developed inhouse
```
sent_analyzer = sentiment_analyzer.SentimentAnalyzer()
```

this part is used by sagemaker for health check of the container. The simplest requirement on the container is to respond with an HTTP 200 status code and an empty body. This indicates to SageMaker that the container is ready to accept inference requests at the /invocations endpoint. Sample code for this ping check is as belows:

```
@app.route('/ping', methods=['GET'])
def ping_check():
    logger.info("PING!")
    return flask.Response(response=json.dumps({"ping_status": "ok"}), status=200)
```

Here is the sample function for predict where the request is parsed and text body is evaluated by the sentiment analyser model.

```
@app.route('/invocations', methods=['POST', 'PUT'])
def predict():
    data = json.loads(flask.request.data.decode('utf-8'))
    prediction = sent_analyzer.predict_proba(data["sentiment"])
    
    response = {'prediction': float(prediction)}
    response = json.dumps(response)
    #resultjson = json.dumps(result)
    return flask.Response(response=response, status=200, mimetype='application/json')
```

For our specific use case the structure of container folder is as follows:

```
         -- sentiment_analyser
            |-- nginx.conf
            |-- predictor.py
            |-- sentiment_analyzer.py
            |--_spu_context_bigru_utils.py
            |-- spu_context_bigru_sentiment.py
            |-- serve
            |-- train
            |-- wsgi.py
            |-- opt
              |-- ml
                |-- model
                  |-- SPUTokenized_word_embedding_16k.matrix
                  |-- SPU_word_tokenizer_16k.model
                  |-- Sentiment_SPUCBiGRU_eval.weights
                  |-- Sentiment_SPUCBiGRU_prod.weights
```

It is important to note that the weights are included in the container. Originally the weights were stored in S3, however in runtime the container doesn't have any outside link. So it is important to keep it self contained. 

#### Adjusting the Dockerfile

Once the container folder is ready we should adjust the dockerfile in the main folder. You may find an sample docker file [here](https://github.com/aws/amazon-sagemaker-examples/blob/main/advanced_functionality/scikit_bring_your_own/container/Dockerfile)
The place that needs to adjust the is the line where we install the python libraries, in our case the line looks like:

```
RUN pip --no-cache-dir install numpy scipy scikit-learn==0.20.2 pandas flask gunicorn tensorboardx  boto3==1.9.243 botocore==1.12.243 tensorflow==2.6.0 regex cyhunspell requests sentencepiece keras==2.6
```

We also should copy the main container directory under /opt/program:

```
COPY vnlp_get_sentiment /opt/program
```

#### Pushing the Docker Image to ECR
The next step is to build the image file and register to AWS ECR. The easiest and hussle free way to do is to use the build_and_push.sh script:
The script to build the Docker image (using the Dockerfile above) and push it to the Amazon EC2 Container Registry (ECR) so that it can be deployed to SageMaker. Specify the name of the image as the argument to this script. The script will generate a full name for the repository in your account and your configured AWS region. If this ECR repository doesn't exist, the script will create it. 
How does the script knows your account configuration? It checks from your aws configuration in your home folder. In case you have more than one active account you should switch to the one that will host you docker image. To do this, run :

```
aws configure
```

The prompt will ask for your confirmation for the access and secret keys. In case you have to switch to another AWS account enter the access and secret keys for that account. 

As next we should run build_and_push.sh script; it takes the image name as argument. Let's say we call the image name 'marketplace-model-container' then we run the build_and_push script as:
```
./build_and_push.sh marketplace-model-container
```
This will download the base image  and customize it further according our instructions defined in the Dockerfile. Once we have it built in our local it will be pushed to ECR in the selected AWS account.


### Testing the image 

### 2- Create the model package in AWS Sagemaker

Once we have the docker in the target account's ECR registered we can build an model package which is the primary requirment for the AWS Marketplace. Even though there are options to launch your product with training capacity, I will only go over the option where our product is fixed and will provide only predictions. See [here](https://github.com/aws/amazon-sagemaker-examples/blob/main/aws_marketplace/creating_marketplace_products/algorithms/Bring_Your_Own-Creating_Algorithm_and_Model_Package.ipynb) for a broader lauch option.

In this part we go over the notebook cells that we will run in AWS Sagemaker:
First go to the Sagemaker - Notebook - Notebook Instances - Create notebook instances.

```
# S3 prefix
prefix = "vnlp-model-serving-marketplace"

import boto3
import re

import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role

# Define IAM role
role = 'arn:aws:iam::707858255059:role/service-role/AmazonSageMaker-ExecutionRole-20230323T120449'

```
Here we have some imports and defining the execution role. 
Running this notebook requires permissions in addition to the normal SageMakerFullAccess permissions. This is because we'll creating new repositories in Amazon ECR. The easiest way to add these permissions is simply to add the managed policy AmazonEC2ContainerRegistryFullAccess to the role that you used to start your notebook instance. There's no need to restart your notebook instance when you do this, the new permissions will be available immediately.


```
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
import json

```
Some needed imports for our usecase, this could be different for another usecase.

```
sess    = boto3.Session()
sm      = sess.client('sagemaker')
region  = sess.region_name
account = boto3.client('sts').get_caller_identity().get('Account')
```


```
import sagemaker
sagemaker_session = sagemaker.Session(boto_session=sess)
```


here we are entering the URI for our uploaded container in ECR
```
image = '707858255059.dkr.ecr.eu-west-1.amazonaws.com/vnlp-sentiment:latest'
```


Some boilerplate code for the configuration. Note for the commented out instance list. This was becuase some of the instances required special quota increase from the AWS. To keep things simple we went only ml.c4.2xlarge.

```
class InferenceSpecification:

    template = """
{    
    "InferenceSpecification": {
        "Containers" : [{"Image": "IMAGE_REPLACE_ME"}],
        "SupportedTransformInstanceTypes": INSTANCES_REPLACE_ME,
        "SupportedRealtimeInferenceInstanceTypes": INSTANCES_REPLACE_ME,\
        "SupportedContentTypes": CONTENT_TYPES_REPLACE_ME,
        "SupportedResponseMIMETypes": RESPONSE_MIME_TYPES_REPLACE_ME
    }
}
"""

    def get_inference_specification_dict(self, ecr_image, supports_gpu, supported_content_types=None, supported_mime_types=None):
        return json.loads(self.get_inference_specification_json(ecr_image, supports_gpu, supported_content_types, supported_mime_types))

    def get_inference_specification_json(self, ecr_image, supports_gpu, supported_content_types=None, supported_mime_types=None):
        if supported_mime_types is None:
            supported_mime_types = []
        if supported_content_types is None:
            supported_content_types = []
        return self.template.replace("IMAGE_REPLACE_ME", ecr_image) \
            .replace("INSTANCES_REPLACE_ME", self.get_supported_instances(supports_gpu)) \
            .replace("CONTENT_TYPES_REPLACE_ME", json.dumps(supported_content_types)) \
            .replace("RESPONSE_MIME_TYPES_REPLACE_ME", json.dumps(supported_mime_types)) \

    def get_supported_instances(self, supports_gpu):
        cpu_list = ["ml.c4.2xlarge"] #"ml.m4.2xlarge","ml.m4.4xlarge","ml.m4.10xlarge","ml.m4.16xlarge","ml.m5.large","ml.m5.xlarge","ml.m5.2xlarge","ml.m5.4xlarge","ml.m5.12xlarge","ml.m5.24xlarge","ml.c4.xlarge","ml.c4.2xlarge","ml.c4.4xlarge","ml.c4.8xlarge","ml.c5.xlarge","ml.c5.2xlarge","ml.c5.4xlarge","ml.c5.9xlarge","ml.c5.18xlarge"]
       # gpu_list = ["ml.p2.xlarge", "ml.p2.8xlarge", "ml.p2.16xlarge", "ml.p3.2xlarge", "ml.p3.8xlarge", "ml.p3.16xlarge"]

        list_to_return = cpu_list

        if supports_gpu:
            list_to_return = cpu_list + gpu_list

        return json.dumps(list_to_return)

```


here we are defining the model package inference configuration

```

modelpackage_inference_specification = InferenceSpecification().get_inference_specification_dict(
    ecr_image=image,
    supports_gpu=False,
    supported_content_types=["application/json", "application/json"],
    supported_mime_types=["application/json"])

```

here we are defining validation specification:

```
class ModelPackageValidationSpecification:
    template = """
{    
    "ValidationSpecification": {
        "ValidationRole": "ROLE_REPLACE_ME",
        "ValidationProfiles": [
            {
                "ProfileName": "ValidationProfile1",
                "TransformJobDefinition": {
                    "MaxConcurrentTransforms": 1,
                    "MaxPayloadInMB": 6,
                    "TransformInput": {
                        "DataSource": {
                            "S3DataSource": {
                                "S3DataType": "S3Prefix",
                                "S3Uri": "BATCH_S3_INPUT_REPLACE_ME"
                            }
                        },
                        "ContentType": "INPUT_CONTENT_TYPE_REPLACE_ME",
                        "CompressionType": "None"
                    },
                    "TransformOutput": {
                        "S3OutputPath": "VALIDATION_S3_OUTPUT_REPLACE_ME/batch-transform-output",
                        "Accept": "OUTPUT_CONTENT_TYPE_REPLACE_ME",
                        "KmsKeyId": ""
                    },
                    "TransformResources": {
                        "InstanceType": "INSTANCE_TYPE_REPLACE_ME",
                        "InstanceCount": 1
                    }
                }
            }
        ]
    }
}    
"""

    def get_validation_specification_dict(self, validation_role, batch_transform_input, input_content_type, output_content_type, instance_type, output_s3_location):
        return json.loads(self.get_validation_specification_json(validation_role, batch_transform_input, input_content_type, output_content_type, instance_type, output_s3_location))

    def get_validation_specification_json(self, validation_role, batch_transform_input, input_content_type, output_content_type, instance_type, output_s3_location):

        return self.template.replace("ROLE_REPLACE_ME", validation_role)\
            .replace("BATCH_S3_INPUT_REPLACE_ME", batch_transform_input)\
            .replace("INPUT_CONTENT_TYPE_REPLACE_ME", input_content_type)\
            .replace("OUTPUT_CONTENT_TYPE_REPLACE_ME", output_content_type)\
            .replace("INSTANCE_TYPE_REPLACE_ME", instance_type)\
            .replace("VALIDATION_S3_OUTPUT_REPLACE_ME", output_s3_location)
```

Validation Specification
In order to provide confidence to the sellers (and buyers) that the products work in Amazon SageMaker before listing them on AWS Marketplace, SageMaker needs to perform basic validations. The product can be listed in AWS Marketplace only if this validation process succeeds. This validation process uses the validation profile and sample data provided by you to run the following validations:

Create a training job in your account to verify your training image works with SageMaker.
Once the training job completes successfully, create a Model in your account using the algorithm's inference image and the model artifacts produced as part of the training job we ran.
Create a transform job in your account using the above Model to verify your inference image works with SageMaker


```
import time

modelpackage_validation_specification = ModelPackageValidationSpecification().get_validation_specification_dict(
    validation_role = role,
    batch_transform_input = "s3://vnlp-model-package-marketplace/backup_vnlp_sentiment_test.json",
    input_content_type = "application/json",
    output_content_type = "application/json",
    instance_type = "ml.c4.2xlarge",
    output_s3_location = 's3://vnlp-model-package-marketplace/')

print(json.dumps(modelpackage_validation_specification, indent=4, sort_keys=True))

```







and finally we are building the main model package for our aws marketplace product:
```
sm_model_name = 'vnlp-sentiment-validated'
model_package_name = sm_model_name + "-" + str(round(time.time()))
create_model_package_input_dict = {
    "ModelPackageName" : model_package_name,
    "ModelPackageDescription" : "model package for vnlp sentiment analysis",
    "CertifyForMarketplace" : True
}
create_model_package_input_dict.update(modelpackage_inference_specification)
create_model_package_input_dict.update(modelpackage_validation_specification)
print(json.dumps(create_model_package_input_dict, indent=4, sort_keys=True))

sm.create_model_package(\*\*create_model_package_input_dict)
```

Once we have run all the cells in the notebook we will see that there is a new model packege created under  Inference - Marketplace model packages. If it was successfully created and validated, you should be able to select the entity and "Publish new ML Marketplace listing" from SageMaker console.





## 3- Prepare the AWS Marketplace listing and the Sample notebook

Once you reach this it is relatively straightforward, you will follow the AWS page and will provide the necessary information. A simple advide would be to check existing similar products in AWS marketplace for the required documentation. You can check the product page of [VNLP Turkish Sentiment Analyzer](VNLP Turkish Sentiment Analyzer)  
The main task here is to prepare a notebook as a user guide. AWS provides a template for this notebook, that you can find [here](https://github.com/aws/amazon-sagemaker-examples/tree/main/aws_marketplace/curating_aws_marketplace_listing_and_sample_notebook/ModelPackage/Sample_Notebook_Template) AWS also give guidance how to customize this notebook [here](https://github.com/aws/amazon-sagemaker-examples/tree/main/aws_marketplace/curating_aws_marketplace_listing_and_sample_notebook/ModelPackage)  
Since this notebook should be public we've created a separate Github repo to host this [notebook](https://github.com/vngrs-ai/marketplace-vnlp/blob/main/VNLP_Turkish_Sentiment_Analyzer.ipynb). As an example how to construct this notebook a walkthrough of the notebook used in our usecase is as follows:


The title of notebook is 'VNLP_Turkish_Sentiment_Analyzer'
A short description of the product should be included:

>Introducing VNLP, the cutting-edge sentiment analysis product that helps you understand the sentiment and emotions behind Turkish text. Our state-of-the-art natural language processing technology analyzes text data quickly and accurately, enabling you to gain insights into your customers’ opinions, attitudes, and emotions. With our intuitive user interface, you can easily upload your Turkish text data and get comprehensive analysis results in seconds. VNGRS is designed with simplicity and ease of use in mind, so you can quickly and easily make data-driven decisions to improve your overall customer experience. Trust VNGRS to take your business to the next level!

then we describe the prerequisites:
>2  Prerequisite
>To run this algorithm you need to have access to the following AWS Services:
>Access to AWS SageMaker and the model package.
>An S3 bucket to specify input/output.
>Role for AWS SageMaker to access input/output from S3.


>3  Set up the environment
>Here we specify a bucket to use and the role that will be used for working with SageMaker.

we make some imports here:  

```
# S3 prefix
prefix = "vnlp-model-serving-marketplace"

# Define IAM role
import boto3
import re

import os
import numpy as np
import pandas as pd
from sagemaker import get_execution_role

role = get_execution_role()
```

```
from sagemaker.model import Model
from sagemaker.serializers import JSONSerializer
from sagemaker.deserializers import JSONDeserializer
from sagemaker.predictor import Predictor
import json
from sagemaker import ModelPackage

```

```
import sagemaker as sage
from time import gmtime, strftime
```

```
sagemaker_session = sage.Session()
#bucket = sagemaker_session.default_bucket()
runtime = boto3.client("runtime.sagemaker")
```

This is our model package that will be utilized for the model endpoint that the user will be use as a prediction service

```
modelpackage_arn = 'arn:aws:sagemaker:eu-west-1:707858255059:model-package/vnlp-sentiment-validated-1679584123'
```

```
sentiment_analyses_model = ModelPackage(
    role=role,
    model_package_arn=modelpackage_arn,
    sagemaker_session=sagemaker_session,
)
```

```
endpoint_name = "vnlp-sentiment-analyses-endpoint"
```




User deploy the model with the specified instance, as long as the endpoint is running it will incur hourly charges. 
So it is important to close it once we/user is done with it.

```
predictor_sentiment_analyser = sentiment_analyses_model.deploy(
    1, "ml.c4.2xlarge", endpoint_name=endpoint_name,
    serializer=JSONSerializer(),
    deserializer=JSONDeserializer(),
)
```

a sample for model input for our sentiment analyzer model
```
payload = '{"sentiment": "urunun yeterli kalitede oldugunu dusunmuyorum"}'
```


we get the response as a float indicating positive or negative values between 0 and 1

```
response = runtime.invoke_endpoint(EndpointName=endpoint_name, ContentType="application/json",
                                   Body=payload)
```

```
sent_analyse_result = response["Body"].read().decode()
```


```
print(sent_analyse_result)
```












