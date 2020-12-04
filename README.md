
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


## 4. cartoon-GAN model:
This model is downloaded from the POST [Generate Anime using CartoonGAN and TensorFlow 2.0](https://leemeng.tw/generate-anime-using-cartoongan-and-tensorflow2-en.html).    
And you can train your own model through the author's repo: https://github.com/mnicnc404/CartoonGan-tensorflow      
There are 4 cartoon style pretrained models: hayao, hosoda, paprika, shinkai (they are the 4 famous Japanese cartoon artists' names).
* hayao url: https://unpkg.com/local-tfjs-models@0.0.2/cartoon-GAN/hayao/model.json   
* hosoda url: https://unpkg.com/local-tfjs-models@0.0.2/cartoon-GAN/hosoda/model.json      
* paprika url: https://unpkg.com/local-tfjs-models@0.0.2/cartoon-GAN/paprika/model.json     
* url: https://unpkg.com/local-tfjs-models@0.0.2/cartoon-GAN/shinkai/model.json     

### how to use it
```typescript
import * as tf from '@tensorflow/tfjs';

const STYLE_MODEL_URL_MAP = {
    "hayao": "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/hayao/model.json",   
    "hosoda": "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/hosoda/model.json",
    "paprika": "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/paprika/model.json",
    "shinkai": "https://unpkg.com/local-tfjs-models@0.0.3/cartoon-GAN/shinkai/model.json"   
};

type STYLE_TYPE = "hayao" | "hosoda" | "paprika" | "shinkai";

async function setupModel(style: STYLE_TYPE): Promise<tf.GraphModel> {
    const model = await tf.loadGraphModel(STYLE_MODEL_URL_MAP[style]);
    // this predict action is used to save time, because every first predict action is very slow.
    model.predict(tf.zeros([1, 1, 1, 3]));
    return model;
}

function predict(
    model: tf.GraphModel, 
    inputImgElement: HTMLImageElement | HTMLCanvasElement, 
    outputElement: HTMLCanvasElement
) {
    // convert the input image to tensor accepted by model
    let inputTensor = tf.browser.fromPixels(inputImgElement);
    inputTensor = inputTensor.toFloat();
    inputTensor = inputTensor.reverse(2);
    inputTensor = tf.expandDims(inputTensor, 0);

    let outputTensor = model.predict(inputTensor as tf.Tensor<tf.Rank>) as tf.Tensor<tf.Rank>;
    // convert the predict tensor to output image
    outputTensor = tf.squeeze(outputTensor, [0]);
    outputTensor = outputTensor.reverse(2);
    outputTensor = outputTensor.mul(0.5).add(0.5);
    outputTensor = tf.clipByValue(outputTensor, 0, 1);

    // put the result into canvas element.
    tf.browser.toPixels(outputTensor as tf.Tensor2D, outputElement);

}

// main
/*
*   style: the name of style model
*   img: the input image/canvas element
*   out: the output canvas element
*/

// load and init model
const model = await setupModel(style = "hayao");
// predict the result and paint it to the output canvas element
predict(model, img, out);

```

