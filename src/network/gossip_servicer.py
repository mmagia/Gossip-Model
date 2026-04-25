import io
import torch
import gossip_pb2
import gossip_pb2_grpc

#Class which uses gRPC methods to provide a way for nodes to communicate over the network
class GossipServicer(gossip_pb2_grpc.GossipNodeServicer):
    def __init__(self, node_instance):
        self.node = node_instance

    def ExchangeWeights(self, request, context):

        incoming_bytes = request.model_weights
        buffer_in = io.BytesIO(incoming_bytes)
        peer_weights_dict = torch.load(buffer_in, weights_only=False)
        peer_accuracy = request.accuracy

        averaged_weights = self.node.handle_incoming_gossip(peer_weights_dict, peer_accuracy)

        buffer_out = io.BytesIO()
        torch.save(averaged_weights, buffer_out)
        outgoing_bytes = buffer_out.getvalue()

        return gossip_pb2.ExchangeReply(
            success=True,
            averaged_weights=outgoing_bytes
        )
