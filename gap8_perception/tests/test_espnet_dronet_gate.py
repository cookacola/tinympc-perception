import torch

from gap8_perception.model_espnet_dronet_gate import ESPNetDroNetGate, deployment_graphs


def test_graph_partition_matches_float_model():
    torch.manual_seed(3)
    model = ESPNetDroNetGate().eval()
    frames = torch.rand(2, 2, 160, 160)
    graphs = deployment_graphs(model)
    with torch.no_grad():
        middle = graphs["encoder"](frames)
        full = model(frames)
        assert torch.equal(graphs["corner_head"](middle), full["corners_raw"])
        assert torch.equal(graphs["gate_head"](middle), full["gate_raw"])
        assert torch.equal(graphs["presence_head"](middle).squeeze(1), full["presence_logit"])
        assert torch.equal(graphs["navigation_head"](middle), full["navigation_logits"])


def test_deployment_shapes():
    model = ESPNetDroNetGate().eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 2, 160, 160))
    assert output["corners_raw"].shape == (1, 4, 20, 20)
    assert output["gate_raw"].shape == (1, 1, 20, 20)
    assert output["presence_logit"].shape == (1,)
    assert output["navigation_logits"].shape == (1, 2)
