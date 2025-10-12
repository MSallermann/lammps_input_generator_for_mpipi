from . import step_widget


class ParseFileWidget(step_widget.StepWidget):
    def __init__(self):
        super().__init__(
            title="Parse File", description="Parse a cif or Pdb file to coarse grain it"
        )
