import cv2
import onnx
import torch
import model


def main():
    # load pre-trained model -------------------------------------------------------------------------------------------
    reid_model = model.build_model()

    # preprocessing stage ----------------------------------------------------------------------------------------------
    input = torch.randn(1, 3, 128, 256, device='cuda')

    # convert to ONNX --------------------------------------------------------------------------------------------------
    ONNX_FILE_PATH = "reid.onnx"
    torch.onnx.export(reid_model,
                      input,
                      ONNX_FILE_PATH,
                      input_names=["input"],
                      output_names=["output"],
                      export_params=True)

    onnx_model = onnx.load(ONNX_FILE_PATH)
    # check that the model converted fine
    onnx.checker.check_model(onnx_model)

    print("Model was successfully converted to ONNX format.")
    print("It was saved to", ONNX_FILE_PATH)


if __name__ == '__main__':
    main()