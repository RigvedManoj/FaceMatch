from flask_ml.flask_ml_server import MLServer
from flask_ml.flask_ml_server.constants import DataTypes
from flask_ml.flask_ml_server.models import (FileInput, ImageResult,
                                             ResponseModel, TextResult)

from src.facematch.interface import FaceMatchModel
from src.facematch.logger import log_info

# create an instance of the model
face_match_model = FaceMatchModel()

# Create a server
server = MLServer(__name__)


# Create an endpoint
@server.route("/findface", DataTypes.IMAGE)
def find_face_endpoint(inputs: list[FileInput], parameters: dict):
    input_file_paths = [item.file_path for item in inputs]
    results = face_match_model.find_face(input_file_paths[0])
    log_info(results)
    image_results = [ImageResult(file_path=res, result=res) for res in results]
    response = ResponseModel(results=image_results)
    return response.get_response()


# Create an endpoint
@server.route("/bulkupload", DataTypes.IMAGE)
def bulk_upload_endpoint(inputs: list[FileInput], parameters: dict):
    input_directory_paths = [item.file_path for item in inputs]
    log_info(input_directory_paths[0])
    result = face_match_model.bulk_upload(input_directory_paths[0])
    text_results = [TextResult(text=result, result=result)]
    response = ResponseModel(results=text_results)
    return response.get_response()


server.run()
