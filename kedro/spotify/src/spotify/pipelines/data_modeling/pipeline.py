from kedro.pipeline import Pipeline, pipeline, node

from .nodes import split_data, scale_data, train_model, scaler_test, evaluate_model


def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=split_data,
                inputs=["model_input_table", "params:model_options"],
                outputs=["X_train", "X_test", "y_train", "y_test"],
                name="split_data_node",
            ),
            node(
                func=scale_data,
                inputs=["X_train", "y_train"],
                outputs=["X_train_scaled", "y_train_scaled", "scaler_x", "scaler_y"],
                name="scale_data_node",
            ),
            node(
                func=train_model,
                inputs=["X_train_scaled", "y_train_scaled", "params:model_options"],
                outputs="regressor",
                name="train_model_node",
            ),
            node(
                func=scaler_test,
                inputs=["X_test", "y_test", "scaler_x", "scaler_y"],
                outputs=["X_test_scaled", "y_test_scaled"],
                name="scaler_test_node",
            ),
            node(
                func=evaluate_model,
                inputs=["regressor", "X_test_scaled", "y_test_scaled"],
                outputs=None,
                name="evaluate_model_node",
            ),
        ]
    )
