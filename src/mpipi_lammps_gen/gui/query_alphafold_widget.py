from . import step_widget


class QueryAlphaFoldWidget(step_widget.StepWidget):
    def __init__(self):
        super().__init__(
            title="QueryAlphaFold",
            description="Query the  alpha fold DB with a uniprot ID",
        )
