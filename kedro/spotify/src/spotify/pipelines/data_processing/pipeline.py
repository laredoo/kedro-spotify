from kedro.pipeline import Pipeline, pipeline, node

from .nodes import first_process_songs, create_model_input_table


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=first_process_songs,
                inputs="songs",
                outputs="songs_first_processed",
                name="first_process_songs_node",
            ),
            node(
                func=create_model_input_table,
                inputs="songs_first_processed",
                outputs="model_input_table",
                name="create_model_input_table_node",
            ),
        ]
    )
