"""
P2P Network Layer
Handles peer-to-peer communication and networking
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Set, Callable
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    """Information about a network peer"""
    peer_id: str
    address: str
    port: int
    last_seen: datetime
    reputation_score: float
    capabilities: List[str]  # validator, trainer, storage, etc.

@dataclass
class NetworkMessage:
    """Network message structure"""
    message_id: str
    message_type: str
    sender: str
    recipient: str  # Can be "broadcast" for broadcast messages
    payload: Dict
    timestamp: datetime

class P2PNetwork:
    """Peer-to-peer network implementation"""
    
    def __init__(self, config):
        self.config = config
        self.node_id = config.network.node_id
        self.listen_port = config.network.listen_port
        self.peers: Dict[str, PeerInfo] = {}
        self.connected_peers: Set[str] = set()
        self.message_handlers: Dict[str, Callable] = {}
        self.server = None
        
        # Register default message handlers
        self.register_message_handlers()
        
    async def start(self):
        """Start the P2P network"""
        logger.info(f"Starting P2P network on port {self.listen_port}")
        
        # Start server to listen for incoming connections
        self.server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.listen_port
        )
        
        # Connect to bootstrap peers
        await self.connect_to_bootstrap_peers()
        
        logger.info(f"P2P network started with node ID: {self.node_id}")
        
    async def connect_to_bootstrap_peers(self):
        """Connect to bootstrap peers"""
        if not self.config.network.bootstrap_peers:
            logger.info("No bootstrap peers configured")
            return
            
        for peer_address in self.config.network.bootstrap_peers:
            try:
                host, port = peer_address.split(':')
                await self.connect_to_peer(host, int(port))
            except Exception as e:
                logger.warning(f"Failed to connect to bootstrap peer {peer_address}: {e}")
                
    async def connect_to_peer(self, host: str, port: int) -> bool:
        """Connect to a peer"""
        try:
            logger.info(f"Connecting to peer {host}:{port}")
            
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send handshake
            handshake = NetworkMessage(
                message_id=f"handshake_{self.node_id}_{datetime.now().timestamp()}",
                message_type="handshake",
                sender=self.node_id,
                recipient="peer",
                payload={
                    "node_id": self.node_id,
                    "capabilities": ["validator", "trainer"],
                    "protocol_version": "0.7"
                },
                timestamp=datetime.now()
            )
            
            await self.send_message_to_writer(writer, handshake)
            
            # Handle peer connection
            peer_id = f"{host}:{port}"
            self.connected_peers.add(peer_id)
            
            # Start message handling for this peer
            asyncio.create_task(self.handle_peer_messages(reader, writer, peer_id))
            
            logger.info(f"Connected to peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to peer {host}:{port}: {e}")
            return False
            
    async def handle_client(self, reader, writer):
        """Handle incoming client connection"""
        peer_address = writer.get_extra_info('peername')
        logger.info(f"New peer connection from {peer_address}")
        
        peer_id = f"{peer_address[0]}:{peer_address[1]}"
        self.connected_peers.add(peer_id)
        
        try:
            await self.handle_peer_messages(reader, writer, peer_id)
        except Exception as e:
            logger.error(f"Error handling peer {peer_id}: {e}")
        finally:
            self.connected_peers.discard(peer_id)
            writer.close()
            await writer.wait_closed()
            
    async def handle_peer_messages(self, reader, writer, peer_id: str):
        """Handle messages from a specific peer"""
        try:
            while True:
                # Read message length
                length_data = await reader.read(4)
                if not length_data:
                    break
                    
                message_length = int.from_bytes(length_data, byteorder='big')
                
                # Read message data
                message_data = await reader.read(message_length)
                if not message_data:
                    break
                    
                # Parse message
                try:
                    message_dict = json.loads(message_data.decode('utf-8'))
                    message = NetworkMessage(
                        message_id=message_dict["message_id"],
                        message_type=message_dict["message_type"],
                        sender=message_dict["sender"],
                        recipient=message_dict["recipient"],
                        payload=message_dict["payload"],
                        timestamp=datetime.fromisoformat(message_dict["timestamp"])
                    )
                    
                    # Handle message
                    await self.handle_message(message, peer_id)
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse message from {peer_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in peer message handling for {peer_id}: {e}")
            
    async def send_message_to_writer(self, writer, message: NetworkMessage):
        """Send message to a specific writer"""
        try:
            # Serialize message
            message_dict = {
                "message_id": message.message_id,
                "message_type": message.message_type,
                "sender": message.sender,
                "recipient": message.recipient,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat()
            }
            
            message_data = json.dumps(message_dict).encode('utf-8')
            message_length = len(message_data)
            
            # Send length prefix and message
            writer.write(message_length.to_bytes(4, byteorder='big'))
            writer.write(message_data)
            await writer.drain()
            
        except Exception as e:
            logger.error(f"Error sending message: {e}")
            
    async def broadcast_message(self, message: NetworkMessage):
        """Broadcast message to all connected peers"""
        logger.info(f"Broadcasting message {message.message_type} to {len(self.connected_peers)} peers")
        
        # Save broadcast message to output
        await self.save_message_to_file(message, "broadcast")
        
        # In a real implementation, this would send to all connected peers
        # For now, just log the broadcast
        
    async def handle_message(self, message: NetworkMessage, sender_peer_id: str):
        """Handle received message"""
        logger.info(f"Received {message.message_type} message from {sender_peer_id}")
        
        # Save received message to output
        await self.save_message_to_file(message, "received")
        
        # Call appropriate message handler
        if message.message_type in self.message_handlers:
            handler = self.message_handlers[message.message_type]
            await handler(message, sender_peer_id)
        else:
            logger.warning(f"No handler for message type: {message.message_type}")
            
    def register_message_handlers(self):
        """Register message handlers"""
        self.message_handlers = {
            "handshake": self.handle_handshake,
            "peer_list": self.handle_peer_list,
            "block_announcement": self.handle_block_announcement,
            "transaction": self.handle_transaction,
            "training_task": self.handle_training_task,
            "training_submission": self.handle_training_submission,
            "wtm_update": self.handle_wtm_update
        }
        
    async def handle_handshake(self, message: NetworkMessage, sender_peer_id: str):
        """Handle handshake message"""
        payload = message.payload
        remote_node_id = payload.get("node_id")
        capabilities = payload.get("capabilities", [])
        
        logger.info(f"Handshake from {remote_node_id} with capabilities: {capabilities}")
        
        # Add peer to known peers only if we have a valid node_id
        if remote_node_id:
            peer_info = PeerInfo(
                peer_id=str(remote_node_id),
                address=sender_peer_id.split(':')[0],
                port=int(sender_peer_id.split(':')[1]),
                last_seen=datetime.now(),
                reputation_score=1.0,
                capabilities=capabilities
            )
            
            self.peers[str(remote_node_id)] = peer_info
        
    async def handle_peer_list(self, message: NetworkMessage, sender_peer_id: str):
        """Handle peer list message"""
        peer_list = message.payload.get("peers", [])
        logger.info(f"Received peer list with {len(peer_list)} peers")
        
        # Process peer list and potentially connect to new peers
        for peer_data in peer_list:
            if peer_data["peer_id"] not in self.peers:
                # Consider connecting to this new peer
                pass
                
    async def handle_block_announcement(self, message: NetworkMessage, sender_peer_id: str):
        """Handle block announcement"""
        block_hash = message.payload.get("block_hash")
        block_number = message.payload.get("block_number")
        
        logger.info(f"Block announcement: #{block_number} ({block_hash})")
        
    async def handle_transaction(self, message: NetworkMessage, sender_peer_id: str):
        """Handle transaction message"""
        transaction_data = message.payload
        logger.info(f"Received transaction: {transaction_data.get('tx_id')}")
        
    async def handle_training_task(self, message: NetworkMessage, sender_peer_id: str):
        """Handle training task message"""
        task_data = message.payload
        logger.info(f"Received training task: {task_data.get('task_id')}")
        
    async def handle_training_submission(self, message: NetworkMessage, sender_peer_id: str):
        """Handle training submission message"""
        submission_data = message.payload
        logger.info(f"Received training submission: {submission_data.get('submission_id')}")
        
    async def handle_wtm_update(self, message: NetworkMessage, sender_peer_id: str):
        """Handle World Truth Model update"""
        update_data = message.payload
        logger.info(f"Received WTM update: {update_data.get('update_type')}")
        
    async def maintain_connections(self):
        """Maintain network connections"""
        # Check peer health
        current_time = datetime.now()
        disconnected_peers = []
        
        for peer_id, peer_info in self.peers.items():
            time_since_last_seen = current_time - peer_info.last_seen
            if time_since_last_seen.total_seconds() > 300:  # 5 minutes timeout
                disconnected_peers.append(peer_id)
                
        # Remove disconnected peers
        for peer_id in disconnected_peers:
            if peer_id in self.peers:
                del self.peers[peer_id]
                self.connected_peers.discard(peer_id)
                logger.info(f"Removed disconnected peer: {peer_id}")
                
        # Try to maintain minimum number of connections
        if len(self.connected_peers) < 3:  # Minimum 3 peers
            await self.discover_and_connect_peers()
            
    async def discover_and_connect_peers(self):
        """Discover and connect to new peers"""
        # In a real implementation, this would use peer discovery mechanisms
        logger.info("Attempting to discover new peers...")
        
    async def save_message_to_file(self, message: NetworkMessage, direction: str):
        """Save message to output file"""
        try:
            message_data = {
                "direction": direction,
                "message_id": message.message_id,
                "message_type": message.message_type,
                "sender": message.sender,
                "recipient": message.recipient,
                "payload": message.payload,
                "timestamp": message.timestamp.isoformat()
            }
            
            filename = f"output/network_message_{message.message_id}.json"
            with open(filename, 'w') as f:
                json.dump(message_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving message to file: {e}") 