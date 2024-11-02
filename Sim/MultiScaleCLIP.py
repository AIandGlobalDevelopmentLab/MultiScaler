import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor, set_seed

class MultiScaleCLIP(nn.Module):
    def __init__(self, vision_model1, vision_model2, visual_projection1, visual_projection2, dropout_prob=0.5):
        super(MultiScaleCLIP, self).__init__()
        self.vision_model1 = vision_model1 # 768
        self.vision_model2 = vision_model2
        self.visual_projection1 = visual_projection1 # from 768 to 512
        self.visual_projection2 = visual_projection2
        
    def get_image_features(self, pixel_values1, pixel_values2):
        vision_outputs1 = self.vision_model1(pixel_values=pixel_values1)
        vision_outputs2 = self.vision_model2(pixel_values=pixel_values2)
        
        image_embeds1 = vision_outputs1.pooler_output
        image_embeds2 = vision_outputs2.pooler_output

        image_embeds1 = self.visual_projection1(image_embeds1)
        image_embeds2 = self.visual_projection2(image_embeds2)
        
        combined_embeds = torch.cat((image_embeds1, image_embeds2), dim=1)

        return combined_embeds
        

    def forward(self, pixel_values1, pixel_values2):
        combined_embeds = self.get_image_representations(pixel_values1, pixel_values2)
        return regression_output

def get_MultiScaleCLIP(UnfreezeTransformer=False, seed=42):
    set_seed(seed)
    model1 = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")
    model2 = CLIPModel.from_pretrained("flax-community/clip-rsicd-v2")  

    for param in model2.parameters():
        param.requires_grad = False

    for param in model1.parameters():
        param.requires_grad = False
    

    vision_model1 = model1.vision_model
    vision_model2 = model2.vision_model  
    visual_projection1 = model1.visual_projection
    visual_projection2 = model2.visual_projection
    
    if UnfreezeTransformer:
        unfreeze_layers = []
        for vision_model in [vision_model1, vision_model2]:
            unfreeze_layers.extend([vision_model.encoder.layers[-1], vision_model.post_layernorm, model1.visual_projection, model2.visual_projection])

        for layer in unfreeze_layers:
            for param in layer.parameters():
                param.requires_grad = True

    MultiCLIP = MultiScaleCLIP(vision_model1, vision_model2, visual_projection1, visual_projection2)

    return MultiCLIP

def get_CLIPProcessor():
    processor = CLIPProcessor.from_pretrained("flax-community/clip-rsicd-v2")
    return processor