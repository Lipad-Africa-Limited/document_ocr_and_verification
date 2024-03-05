from fastapi import FastAPI, status 
from typing import Dict, Any, Optional, Union, List, Tuple, cast
from pydantic import BaseModel, RootModel
from fastapi import HTTPException 
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from abc import ABC, abstractmethod
from PIL import Image
import os
import base64
# import service
import easyocr
import keras_ocr
import pandas as pd
import cv2
import numpy as np
import json
import uuid
import requests
from numpy.typing import NDArray


app = FastAPI()

# dataclasses
# requests
class OcrRequest(BaseModel):
    request_ref: str 
    document_ref: str # can have same doc diff ref where in such a case u may evn give null on image i sps as no need to reimport for instanc eor jst work on same folder evn if irimporting ie hnc folder is doc/req_ref/things
    document_type: Optional[str] # not used for now but later we could use to load specific finetuned models for particular tasks eg numbers etc
    
class Base64OcrRequest(OcrRequest):
    image_base64: str # backend serving, static file server or s3
    
class UrlOcrRequest(OcrRequest):
    image_url: str # frm a shared local folder
    
class FileOcrRequest(OcrRequest):
    image_path: str 
    
# EitherOcrRequest = RootModel[Union[Base64OcrRequest, UrlOcrRequest, FileOcrRequest]]
EitherOcrRequest = Union[Base64OcrRequest, UrlOcrRequest, FileOcrRequest] # this one i loose validation though i can cast so instead use bove and do pattenr matching??? seem to work though ok so wl use this todo 11111111111 review later

# response
class OcrToken(BaseModel):
    bbox: Any # edit further later
    word: str
    accepted: bool # simple user
    score: float # advanced user
    # foreground: str # used later
    # background: str # used later
    # def __init__(self,,bbox,word,accepted,score):
    #   self.name = name
    #   self.age = age
    
class OcrResponse(BaseModel):
    request_ref: str
    document_ref: str
    ocr_tokens: List [ OcrToken ]
    
# models


    
# controllers : for now i use static methods later we may se if it is necesary to objectize
# toplevel
class BaseController:
    def send_success(self, data: OcrResponse, message: str="Request Successful", status: str="success", status_code: status = status.HTTP_200_OK) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=jsonable_encoder({"status": status,"message": message, "data": data}),
        ) 

    def send_error(self, request: BaseModel, data: Any=None, message: str ="Request failed", status: str="failed", status_code: status = status.HTTP_409_CONFLICT) -> JSONResponse:
        return JSONResponse(
            status_code=status_code,
            content=jsonable_encoder({"status": status,"message": message, "data": data, "data":request}),
        ) 
        
class OcrControllerInterface(ABC):
    @abstractmethod
    async def recognize(self, request: OcrRequest, image: Image.Image, image_np: NDArray) -> JSONResponse:
        pass # i specify ocrrequest as once image is extracted we do not need specific res
    
# bottomlevel
class StatusController( BaseController ):
    def get_status(self, request: BaseModel) -> Dict[str, Any]:
        data = {'status': 'ok'}
        return self.send_success(data, f"Server is running")
        # return self.send_error(request, message=f"Server under maintenance")
    

class EasyOcrController( BaseController, OcrControllerInterface):
    async def recognize(self, request: OcrRequest, image: Image.Image, image_np: NDArray) -> JSONResponse:
        # ocr_tokens=[OcrToken(bbox=[12,342,543], word="word1", accepted=True, score=82.4554), ]
        ocr_tokens = await easyocr_recognize(image,request.document_ref)
        ocr_res=OcrResponse(request_ref=request.request_ref,document_ref=request.document_ref, ocr_tokens=ocr_tokens)
        print(ocr_res)
        return self.send_success(ocr_res, message=f"from image of {image_np.shape} pixels")
class KerasOcrController( BaseController, OcrControllerInterface):
    async def recognize(self, request: OcrRequest, image: Image.Image, image_np: NDArray) -> JSONResponse:
        ocr_tokens = await kerasocr_recognize(image,request.document_ref)
        ocr_res=OcrResponse(request_ref=request.request_ref,document_ref=request.document_ref, ocr_tokens=ocr_tokens)
        print(ocr_res)
        return self.send_success(ocr_res, message=f"from image of {image_np.shape} pixels")
class DefaultErrorController( BaseController, OcrControllerInterface):
    async def recognize(self, request: OcrRequest, message:str =f"Sorry engine selected not found" ) -> JSONResponse:
        return self.send_error(request, message=message)
        


# apis
@app.get('/status')
async def status_check():
    return StatusController().get_status(None)

@app.post('/engine/{engine_code}/recognize')
async def recognize(engine_code: str, ocr_request: EitherOcrRequest) ->JSONResponse:
    # i suppose common routine ie such as loading the image
    if isinstance(ocr_request, Base64OcrRequest):   
        ocr_request=cast(Base64OcrRequest, ocr_request)
        [image_raw,image_np]=decode_base64(ocr_request.image_base64,ocr_request.document_ref)
    elif isinstance(ocr_request, UrlOcrRequest): 
        ocr_request=cast(UrlOcrRequest, ocr_request.image_url)
        [image_raw,image_np]=retrieve_url(ocr_request.image_url,ocr_request.document_ref)
    else: # request=cast(FileOcrRequest, ocr_request) # otther wise just load image from fs
        return await DefaultErrorController().recognize(ocr_request, "Sorry direct file imagepath not supported for now for "+str(type(ocr_request)))
    # for now we manually route as is hard coded but later engine code may load particular model off a db or etc with its configs and all
    if engine_code == "ENGINE_1":
      return await EasyOcrController().recognize(ocr_request, image_raw, image_np)
    elif engine_code == "ENGINE_2":
      return await KerasOcrController().recognize(ocr_request, image_raw, image_np)
    else:
      return await DefaultErrorController().recognize(ocr_request, image_raw, image_np)


# server
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9001)












# decode and init
reader = easyocr.Reader(['en']) # ideally this will be forked out to a single service ie use lifestyle

async def easyocr_recognize(image:Image.Image, reference):
    # print(image);
    result = reader.readtext(image) 
    THRESHOLD=.5 # for now but review later
    ocr_tokens=[ OcrToken(bbox= [[int(j) for j in i]  for i in x[0]] , word=x[1], accepted=x[2]>THRESHOLD, score=x[2]) for x in result]
    # write_json(save_dir, 'original-ocr_tokens.txt', ocr_tokens)
    return ocr_tokens;


pipeline = keras_ocr.pipeline.Pipeline()

async def kerasocr_recognize(image:Image.Image, reference):  
    # print(image); 
    prediction_groups = pipeline.recognize([image])
    # prediction_groups = pipeline.recognize([keras_ocr.tools.read(image)])
    ocr_tokens=[ OcrToken(bbox= [[int(j) for j in i]  for i in x[0]] , word=x[1], accepted=True, score=80.1) for x in prediction_groups[0]]
    # write_json(save_dir, 'original-ocr_tokens.txt', ocr_tokens)
    return ocr_tokens;







# decode and init
def retrieve_url(image_url:str, reference:str) -> Tuple [Image.Image,NDArray]: 
    response = requests.get(image_url, stream=True)
    im_binary = Image.open(response.raw)
    nparr = np.array(im_binary)
    write_image(nparr,reference)
    return im_binary,nparr;

# decode and init
def decode_base64(image_base64:str, reference:str) -> Tuple [Image.Image,NDArray]: 
    im_binary = base64.b64decode(image_base64) 
    nparr = np.fromstring(im_binary, np.uint8)
    write_image(nparr,reference)
    return im_binary,nparr;
    
    



# some image processing needed to scale to fixed size always for performance reasons * for now we dont use
def resize_image(img, size):
   # Get the original image size
   h, w = img.shape[:2]

   # Calculate the aspect ratio of the image
   aspect_ratio = w/h

   # Calculate the new dimensions of the image
   if aspect_ratio > 1: # Image is wider than it is tall
       new_w = size
       new_h = np.round(new_w / aspect_ratio).astype(int)
   elif aspect_ratio < 1: # Image is taller than it is wide
       new_h = size
       new_w = np.round(new_h * aspect_ratio).astype(int)
   else: # Image is a square
       new_h, new_w = size, size

   # Resize the image
   resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

   return resized_img

# caching the findings
def write_json(target_path, target_file, data):
    if not os.path.exists(target_path):
        try:
            os.makedirs(target_path)
        except Exception as e:
            print(e)
            raise
    with open(os.path.join(target_path, target_file), 'w') as f:
        json.dump(data, f)
        
def write_image(nparr: NDArray, reference:str):
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR) 
    cur_path = os.path.dirname(os.path.realpath(__file__))
    save_dir = os.path.join(cur_path, 'images/prod/'+reference )
    os.makedirs(save_dir, exist_ok=True)
    cv2.imwrite(os.path.join(save_dir, 'original.jpg'), img) 
    