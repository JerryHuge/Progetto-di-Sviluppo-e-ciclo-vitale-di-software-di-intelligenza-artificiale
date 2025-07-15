import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.train_mnist import train_mnist, evaluate_mnist

def test_mnist_training():
    test_model_path = "test_mnist_model.pth"
    train_mnist(1, save_path=test_model_path)

    assert os.path.exists(test_model_path), "Model file not found after training."
    assert os.path.getsize(test_model_path) > 0, "Model file is empty."

    import torch
    from src.train_mnist import SimpleCNN
    model = SimpleCNN()
    model.load_state_dict(torch.load(test_model_path))

    xin = torch.randn(1, 1, 28, 28)
    model.eval()
    with torch.no_grad():
        output = model(xin)
    assert output.shape == (1, 10), "Model output shape is incorrect."

    os.remove(test_model_path)
    print("Test passed: Model trained and saved successfully.")

def test_mnist_evaluation():
    test_model_path = "test_mnist_model.pth"
    train_mnist(1, save_path=test_model_path)
    evaluate_mnist(test_model_path)

    os.remove(test_model_path)

    confusion_path = os.path.join("outputs", "confusion_matrix.png")
    if os.path.exists(confusion_path):
        os.remove(confusion_path)


if __name__ == "__main__":
    test_mnist_training()
    test_mnist_evaluation()