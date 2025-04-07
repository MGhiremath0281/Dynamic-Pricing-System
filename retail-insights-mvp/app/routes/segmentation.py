from flask import Blueprint, render_template, request

segmentation_bp = Blueprint('segmentation', __name__)

@segmentation_bp.route('/', methods=['GET', 'POST'])
def segmentation():
    segment = None

    if request.method == 'POST':
        unit_price = float(request.form.get('unit_price'))
        quantity = int(request.form.get('quantity'))
        total = float(request.form.get('total'))
        rating = float(request.form.get('rating'))

        # Call your segmentation model here
        # Example dummy logic:
        segment = "High Value" if total > 500 else "Regular"

    return render_template('segmentation.html', segment=segment)
