"""
P2P Network Layer for WTM
Handles peer-to-peer communication and networking
WTM-specific messaging and knowledge synchronization
"""
import asyncio
import logging
import json
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class PeerInfo:
    """Information about a WTM network peer"""
    peer_id: str
    address: str
    port: int
    last_seen: datetime
    reputation_score: float
    capabilities: List[str]  # validator, trainer, storage, knowledge_provider, etc.
    knowledge_domains: List[str]  # WTM domains this peer contributes to

@dataclass
class NetworkMessage:
    """Network message structure for WTM"""
    message_id: str
    message_type: str
    sender: str
    recipient: str  # Can be "broadcast" for broadcast messages
    payload: Dict
    timestamp: datetime

class P2PNetwork:
    """Peer-to-peer network implementation for WTM"""
    
    def __init__(self, config):
        self.config = config
        self.node_id = config.network.node_id
        self.listen_port = config.network.listen_port
        self.peers: Dict[str, PeerInfo] = {}
        self.connected_peers: Set[str] = set()
        self.message_handlers: Dict[str, Callable] = {}
        self.server = None
        
        # Register WTM-specific message handlers
        self.register_message_handlers()
        
    async def start(self):
        """Start the WTM P2P network"""
        logger.info(f"Starting WTM P2P network on port {self.listen_port}")
        
        # Start server to listen for incoming connections
        self.server = await asyncio.start_server(
            self.handle_client,
            '0.0.0.0',
            self.listen_port
        )
        
        # Connect to bootstrap peers
        await self.connect_to_bootstrap_peers()
        
        logger.info(f"WTM P2P network started with node ID: {self.node_id}")
        
    async def connect_to_bootstrap_peers(self):
        """Connect to WTM bootstrap peers"""
        if not self.config.network.bootstrap_peers:
            logger.info("No WTM bootstrap peers configured")
            return
            
        for peer_address in self.config.network.bootstrap_peers:
            try:
                host, port = peer_address.split(':')
                await self.connect_to_peer(host, int(port))
            except Exception as e:
                logger.warning(f"Failed to connect to WTM bootstrap peer {peer_address}: {e}")
                
    async def connect_to_peer(self, host: str, port: int) -> bool:
        """Connect to a WTM peer"""
        try:
            logger.info(f"Connecting to WTM peer {host}:{port}")
            
            reader, writer = await asyncio.open_connection(host, port)
            
            # Send WTM handshake
            handshake = NetworkMessage(
                message_id=f"wtm_handshake_{self.node_id}_{datetime.now().timestamp()}",
                message_type="wtm_handshake",
                sender=self.node_id,
                recipient="peer",
                payload={
                    "node_id": self.node_id,
                    "capabilities": ["validator", "trainer", "knowledge_provider"],
                    "protocol_version": "WTM_0.8",
                    "knowledge_domains": ["science", "technology", "mathematics"]
                },
                timestamp=datetime.now()
            )
            
            await self.send_message_to_writer(writer, handshake)
            
            # Handle peer connection
            peer_id = f"{host}:{port}"
            self.connected_peers.add(peer_id)
            
            # Start message handling for this peer
            asyncio.create_task(self.handle_peer_messages(reader, writer, peer_id))
            
            logger.info(f"Connected to WTM peer {peer_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to WTM peer {host}:{port}: {e}")
            return False
            
    async def handle_client(self, reader, writer):
        """Handle incoming WTM client connection"""
        peer_address = writer.get_extra_info('peername')
        logger.info(f"New WTM peer connection from {peer_address}")
        
        peer_id = f"{peer_address[0]}:{peer_address[1]}"
        self.connected_peers.add(peer_id)
        
        try:
            await self.handle_peer_messages(reader, writer, peer_id)
        except Exception as e:
            logger.error(f"Error handling WTM peer {peer_id}: {e}")
        finally:
            self.connected_peers.discard(peer_id)
            writer.close()
            await writer.wait_closed()
            
    async def handle_peer_messages(self, reader, writer, peer_id: str):
        """Handle messages from a specific WTM peer"""
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
                    logger.error(f"Failed to parse message from WTM peer {peer_id}: {e}")
                    
        except Exception as e:
            logger.error(f"Error in WTM peer message handling for {peer_id}: {e}")
            
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
            logger.error(f"Error sending WTM message: {e}")
            
    async def broadcast_message(self, message: NetworkMessage):
        """Broadcast message to all connected WTM peers"""
        logger.info(f"Broadcasting WTM message {message.message_type} to {len(self.connected_peers)} peers")
        
        # Save broadcast message to output
        await self.save_message_to_file(message, "broadcast")
        
        # In a real implementation, this would send to all connected peers
        # For now, just log the broadcast
        
    async def handle_message(self, message: NetworkMessage, sender_peer_id: str):
        """Handle received WTM message"""
        logger.info(f"Received WTM {message.message_type} message from {sender_peer_id}")
        
        # Save received message to output
        await self.save_message_to_file(message, "received")
        
        # Call appropriate message handler
        if message.message_type in self.message_handlers:
            handler = self.message_handlers[message.message_type]
            await handler(message, sender_peer_id)
        else:
            logger.warning(f"No handler for WTM message type: {message.message_type}")
            
    def register_message_handlers(self):
        """Register WTM-specific message handlers"""
        self.message_handlers = {
            "wtm_handshake": self.handle_wtm_handshake,
            "peer_list": self.handle_peer_list,
            "block_announcement": self.handle_block_announcement,
            "transaction": self.handle_transaction,
            "training_task": self.handle_training_task,
            "training_submission": self.handle_training_submission,
            "knowledge_update": self.handle_knowledge_update,
            "knowledge_query": self.handle_knowledge_query,
            "knowledge_sync": self.handle_knowledge_sync,
            "domain_update": self.handle_domain_update
        }
        
    async def handle_wtm_handshake(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM handshake message"""
        payload = message.payload
        remote_node_id = payload.get("node_id")
        capabilities = payload.get("capabilities", [])
        knowledge_domains = payload.get("knowledge_domains", [])
        
        logger.info(f"WTM handshake from {remote_node_id} with capabilities: {capabilities}, domains: {knowledge_domains}")
        
        # Add peer to known peers only if we have a valid node_id
        if remote_node_id:
            peer_info = PeerInfo(
                peer_id=str(remote_node_id),
                address=sender_peer_id.split(':')[0],
                port=int(sender_peer_id.split(':')[1]),
                last_seen=datetime.now(),
                reputation_score=1.0,
                capabilities=capabilities,
                knowledge_domains=knowledge_domains
            )
            
            self.peers[str(remote_node_id)] = peer_info
        
    async def handle_peer_list(self, message: NetworkMessage, sender_peer_id: str):
        """Handle peer list message"""
        peer_list = message.payload.get("peers", [])
        logger.info(f"Received WTM peer list with {len(peer_list)} peers")
        
        # Process peer list and potentially connect to new peers
        for peer_data in peer_list:
            if peer_data["peer_id"] not in self.peers:
                # Consider connecting to this new WTM peer
                pass
                
    async def handle_block_announcement(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM block announcement"""
        block_hash = message.payload.get("block_hash")
        block_number = message.payload.get("block_number")
        knowledge_hash = message.payload.get("knowledge_hash")
        
        logger.info(f"WTM block announcement: #{block_number} ({block_hash}) with knowledge: {knowledge_hash}")
        
    async def handle_transaction(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM transaction message"""
        transaction_data = message.payload
        logger.info(f"Received WTM transaction: {transaction_data.get('tx_id')}")
        
    async def handle_training_task(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM training task message"""
        task_data = message.payload
        logger.info(f"Received WTM training task: {task_data.get('task_id')} for domain: {task_data.get('knowledge_target')}")
        
    async def handle_training_submission(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM training submission message"""
        submission_data = message.payload
        logger.info(f"Received WTM training submission: {submission_data.get('submission_id')}")
        
    async def handle_knowledge_update(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM knowledge update message"""
        update_data = message.payload
        update_type = update_data.get("update_type")
        domain = update_data.get("domain")
        
        logger.info(f"Received WTM knowledge update: {update_type} for domain: {domain}")
        
    async def handle_knowledge_query(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM knowledge query message"""
        query_data = message.payload
        query = query_data.get("query")
        domain = query_data.get("domain")
        
        logger.info(f"Received WTM knowledge query: '{query}' in domain: {domain}")
        
        # In a full implementation, this would query the local WTM and respond
        
    async def handle_knowledge_sync(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM knowledge synchronization message"""
        sync_data = message.payload
        domain = sync_data.get("domain")
        entities_count = len(sync_data.get("entities", []))
        facts_count = len(sync_data.get("facts", []))
        
        logger.info(f"Received WTM knowledge sync for domain {domain}: {entities_count} entities, {facts_count} facts")
        
    async def handle_domain_update(self, message: NetworkMessage, sender_peer_id: str):
        """Handle WTM domain update message"""
        domain_data = message.payload
        domain_id = domain_data.get("domain_id")
        completeness = domain_data.get("completeness")
        
        logger.info(f"Received WTM domain update for {domain_id}: completeness {completeness}")
        
    async def broadcast_knowledge_update(self, update_data: Dict):
        """Broadcast a knowledge update to WTM network"""
        message = NetworkMessage(
            message_id=f"wtm_knowledge_{datetime.now().timestamp()}",
            message_type="knowledge_update",
            sender=self.node_id,
            recipient="broadcast",
            payload=update_data,
            timestamp=datetime.now()
        )
        
        await self.broadcast_message(message)
        
    async def request_knowledge_sync(self, domain: str, peer_id: Optional[str] = None):
        """Request knowledge synchronization for a domain"""
        message = NetworkMessage(
            message_id=f"wtm_sync_req_{datetime.now().timestamp()}",
            message_type="knowledge_sync_request",
            sender=self.node_id,
            recipient=peer_id or "broadcast",
            payload={"domain": domain, "last_sync": datetime.now().isoformat()},
            timestamp=datetime.now()
        )
        
        if peer_id:
            # Send to specific peer (would need peer connection)
            logger.info(f"Requesting WTM knowledge sync from {peer_id} for domain {domain}")
        else:
            await self.broadcast_message(message)
            
    async def maintain_connections(self):
        """Maintain WTM network connections"""
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
                logger.info(f"Removed disconnected WTM peer: {peer_id}")
                
        # Try to maintain minimum number of connections
        if len(self.connected_peers) < 3:  # Minimum 3 peers
            await self.discover_and_connect_peers()
            
    async def discover_and_connect_peers(self):
        """Discover and connect to new WTM peers"""
        # In a real implementation, this would use peer discovery mechanisms
        logger.info("Attempting to discover new WTM peers...")
        
    async def get_network_statistics(self) -> Dict[str, Any]:
        """Get WTM network statistics"""
        domain_coverage: Dict[str, int] = {}
        for peer in self.peers.values():
            for domain in peer.knowledge_domains:
                domain_coverage[domain] = domain_coverage.get(domain, 0) + 1
                
        return {
            "connected_peers": len(self.connected_peers),
            "known_peers": len(self.peers),
            "domain_coverage": domain_coverage,
            "avg_reputation": sum(p.reputation_score for p in self.peers.values()) / max(len(self.peers), 1),
            "capabilities_distribution": {}
        }
        
    async def save_message_to_file(self, message: NetworkMessage, direction: str):
        """Save WTM message to output file"""
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
            
            filename = f"output/wtm_network_message_{message.message_id}.json"
            with open(filename, 'w') as f:
                json.dump(message_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Error saving WTM message to file: {e}") 