from mmocr.apis import MMOCRInferencer
import matplotlib.pyplot as plt
import mmcv


infer = MMOCRInferencer(rec='satrn')
result = infer('mmor_demo/demo/demo_text_recog.jpg', save_vis=True, return_vis=True)
print(result['predictions'])
plt.imshow(result['visualization'][0])
plt.savefig('text.png')

# predicted_img = mmcv.imread('results/vis/demo_text_recog.jpg')
# plt.imshow(mmcv.bgr2rgb(predicted_img))
# plt.savefig('img.png')