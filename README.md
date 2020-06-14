
# local-tfjs-models
This is a repo of local open source tfjs-models. 
For some reason you know, we can not fetch models from google storage service in some region of the world.
So we can not directly use the npm: @tensorflow-models.

This is a collection repo of tfjs-models. Then you can deploy these models in your local or cdn enviroment. 


# Available Models from unpkg cdn
## 1. blazeface model:
* url: https://unpkg.com/local-tfjs-models@0.0.1/blazeface/google/model.json      
This model is used to detect face box and keypoints for ears/eyes/nose/mouse.   


## 2. facemesh model:
* url: https://unpkg.com/local-tfjs-models@0.0.1/facemesh/google/model.json
This model is used to detect face mesh.   
Actually you need use blazeface model first to get the face box, then use the facemesh model to detect the face mesh. 
You need to build a simple pipeline to get the final face mesh.   


## 3. imagescore model:
This model is downloaded from bilibili.com. They use this model to score image to decide which image should probally be the cover of the video.     
It is for study usage only. I am not sure whether you can use it freely.      
By the way, trainning a image score model is pretty simple, you can do it on your own dataset.  
* url: https://unpkg.com/local-tfjs-models@0.0.1/imagescore/bilibili/model.json
It returns the score of image, max value is 5, min value is 0. 
Acturally it returns a tensor with 2 elements, and the each element' value is the same, representing the image score.     
