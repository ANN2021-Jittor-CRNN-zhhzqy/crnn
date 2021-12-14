from jittor.utils.pytorch_converter import convert

with open("../torch/model.py", "r", encoding="utf-8") as f:
    pytorch_code = f.read()

jittor_code = convert(pytorch_code)

with open("./model.py", "w", encoding="utf-8") as f1:
    f1.write(jittor_code)
