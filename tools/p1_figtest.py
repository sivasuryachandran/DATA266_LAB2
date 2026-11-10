import os, sys, time
sys.path.insert(0, os.getcwd())
from PIL import Image
from common.vlm import VLM
PROMPT = ("This is a chart from an economics document. 1) Give the title, axis labels and legend. "
          "2) List EVERY numeric label printed on the chart (data labels next to lines or bars), with the series it belongs to and its x position. "
          "3) For each series, read the value at each labelled x tick (state 'approx.' for values you estimate from gridlines). "
          "Be precise; do not invent values. Quote any other text verbatim after TEXT_IN_IMAGE:.")
vlm = VLM(sys.argv[1], quant="4bit")
for name in sys.argv[2:]:
    t0 = time.time()
    out = vlm.generate(PROMPT, images=[Image.open(f"work_p1/figures/{name}").convert("RGB")], max_new_tokens=700)
    print("=====", name, round(time.time() - t0), "s\n", out)
