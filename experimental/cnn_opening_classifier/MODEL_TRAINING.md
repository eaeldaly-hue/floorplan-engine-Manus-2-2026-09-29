# Commercial-use door/window classifier

The old CubiCasa checkpoint is no longer part of the project runtime. Its
source and weights were moved outside the checkout because their license is
non-commercial.

The replacement is trained locally from PERDAW's MIT-licensed plan-view door
and window examples. The classifier only evaluates a candidate patch; wall
gaps and room topology still come from this project's geometry pipeline.

## Train locally

1. Download the `perdaw_object_view_split.zip` archive from the
   [PERDAW Zenodo record](https://zenodo.org/records/10823907) and extract it
   outside Git, for example to `data/perdaw/`.
2. Install PyTorch in the active project environment:
   `pip install 'torch>=2.2,<3'`.
3. From the repository root, run
   `python -m experimental.cnn_opening_classifier.train_opening_symbol_model data/perdaw/`.

The script trains on plan-view images only and holds out complete object
families for validation. It saves the checkpoint to
`experimental/cnn_opening_classifier/models/opening_symbol_classifier.pt`
(the committed checkpoint lives there). No network API is used at inference
time.

PERDAW contains 28,800 rendered plan and elevation views derived from 20
designed door/window objects, so its validation result is a useful symbol check but does not prove
accuracy on arbitrary scanned plans. The classifier must be evaluated against
separate, labeled full-floor-plan examples before being treated as production
quality. The dataset and its license are described in the
[PERDAW repository](https://github.com/alexandru-filip/perdaw-dataset).
The included [PERDAW license notice](third_party/perdaw/LICENSE) preserves the
dataset's attribution terms.
