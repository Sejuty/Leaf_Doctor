# Assets

## sample_leaf.jpg

A tomato sprig showing **Septoria leaf spot** — one of the 15 classes.

- Source: [Tomato septoria leaf spot 3006.jpg](https://commons.wikimedia.org/wiki/File:Tomato_septoria_leaf_spot_3006.jpg), Wikimedia Commons
- Licence: **CC0** (public domain)

Used as a smoke-test image for `app.py` and the notebook.

**Note:** the shipped model *misclassifies* this image (it predicts Late blight
or Yellow Leaf Curl Virus, not Septoria leaf spot). That is expected and is a
useful illustration of the limitation documented in the main README: PlantVillage
consists of single flat leaves on plain backgrounds, whereas this is a field
photo of a whole sprig with a dark background. The model is out of its
distribution here.

If you want a fair demonstration of model accuracy, use an image from the
PlantVillage validation split instead (`data/PlantVillage/` after running
`python -m leafdoctor.data`).
