export onnx is from  jamjamjon/usls
thanks a lot !

1.generate onnx file 

python export_onnx.py --all --model-path ./onnx-models-v2/ --output-dir ./onnx-models-v2/ --image-height 1008 --image-width 1008

2.build trt engine set batch size as you want 

python build_trt.py --onnx-dir onnx-models-v2 --output-dir onnx-models-v2 --opt-batch-size 16 --max-batch-size 16 --min-batch-size 1

3. infer mask

python infer_trt_multi.py --image ./cam0 --texts car tree building --model-dir ./onnx-models-v2/ --tokenizer ./tokenizer.json --output ./output_cam0 --conf 0.5 0.4 0.5 --batch-size 16 --image-height 1008 --image-width 1008
