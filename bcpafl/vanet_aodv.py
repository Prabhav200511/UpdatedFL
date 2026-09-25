"""Modeled multihop AODV overlay for BC-PAFL (ported from ProxyFL v1).

v2's real transport is single-hop V2I, so AODV here is a *modeled* overlay in
exactly v1's sense: per round, the ground-truth topology is frozen and every
FL envelope (uplink beacon/upload, downlink broadcast/config) is submitted to
a destination-only, ideal-link AODV subset (RFC 3561, no intermediate replies,
HELLOs, expanding rings, local repair or secure routing).  Link capacities
come from v2's :mod:`bcpafl.channel` budget; all timing is simulated and never
sleeps.  Produces the same ``*_routing_rounds.csv`` / ``*_metadata.json``
schema as v1.
"""

from __future__ import annotations

import csv
import heapq
import itertools
import json
import math
import threading
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Callable, Dict, List, Optional, Sequence, Tuple

IP_UDP_HEADER_BYTES = 28
RREQ_BODY_BYTES = 24
RREP_BODY_BYTES = 20
SPEED_OF_LIGHT = 299_792_458.0


# ----------------------------------------------------------------------
# AODV protocol (v1 aodv.py, verbatim logic)
# ----------------------------------------------------------------------
def sequence_newer(value, previous):
    return 0 < ((value - previous) & 0xFFFFFFFF) < 0x80000000


@dataclass(frozen=True)
class AodvSettings:
    active_route_timeout: float = 3.0
    node_traversal_time: float = 0.04
    network_diameter: int = 64
    rreq_retries: int = 2
    packet_payload_bytes: int = 1200

    @property
    def net_traversal_time(self):
        return 2 * self.node_traversal_time * self.network_diameter

    @property
    def path_discovery_time(self):
        return 2 * self.net_traversal_time


@dataclass
class Route:
    destination: str
    next_hop: str
    hop_count: int
    sequence: int
    expiry: float
    sequence_valid: bool = True
    valid: bool = True
    precursors: set = field(default_factory=set)


class AodvProtocol:
    def __init__(self, simulator, settings):
        self.sim = simulator
        self.settings = settings
        self.tables: Dict[str, Dict[str, Route]] = {}
        self.sequences: Dict[str, int] = {}
        self.request_ids: Dict[str, int] = {}
        self.seen: Dict[tuple, float] = {}

    def install(self, node, destination, next_hop, hops, sequence, expiry):
        table = self.tables.setdefault(node, {})
        old = table.get(destination)
        fresh = old is None or not old.sequence_valid or sequence_newer(sequence, old.sequence)
        equal_better = old is not None and sequence == old.sequence and (
            not old.valid or hops < old.hop_count)
        if fresh or equal_better:
            table[destination] = Route(destination, next_hop, hops, sequence,
                                       expiry, precursors=set() if old is None
                                       else old.precursors.copy())
        elif old.sequence == sequence and old.next_hop == next_hop:
            old.expiry = max(old.expiry, expiry)
        return table[destination]

    def active(self, node, destination):
        route = self.tables.get(node, {}).get(destination)
        if route is not None and route.expiry <= self.sim.now:
            route.valid = False
        return route if route is not None and route.valid else None

    def discover(self, source, destination):
        old = self.tables.get(source, {}).get(destination)
        if old is not None:
            old.valid = False
        for attempt in range(self.settings.rreq_retries + 1):
            start = self.sim.now
            self.seen = {k: e for k, e in self.seen.items() if e > start}
            self.sequences[source] = (self.sequences.get(source, 0) + 1) & 0xFFFFFFFF
            self.request_ids[source] = (self.request_ids.get(source, 0) + 1) & 0xFFFFFFFF
            old = self.tables.get(source, {}).get(destination)
            request = dict(origin=source, destination=destination,
                           request_id=self.request_ids[source],
                           origin_sequence=self.sequences[source],
                           destination_sequence=old.sequence if old and old.sequence_valid else None,
                           hops=0, ttl=self.settings.network_diameter)
            self.seen[(source, source, request["request_id"])] = (
                start + self.settings.path_discovery_time)
            self.sim.control("RREQ", source, None, RREQ_BODY_BYTES, request,
                             self.receive_request)
            deadline = start + self.settings.net_traversal_time * (2 ** attempt)
            self.sim.drain(until=lambda: self.active(source, destination) is not None,
                           deadline=deadline)
            if self.active(source, destination):
                return True
            self.sim.now = max(self.sim.now, deadline)
        return False

    def receive_request(self, node, previous, request):
        key = (node, request["origin"], request["request_id"])
        if self.seen.get(key, -1) > self.sim.now:
            return
        self.seen[key] = self.sim.now + self.settings.path_discovery_time
        hops = request["hops"] + 1
        self.install(node, request["origin"], previous, hops, request["origin_sequence"],
                     self.sim.now + max(self.settings.active_route_timeout,
                                        self.settings.path_discovery_time))
        if node == request["destination"]:
            sequence = self.sequences.get(node, 0)
            requested = request["destination_sequence"]
            if requested is not None and (requested == sequence
                                          or sequence_newer(requested, sequence)):
                sequence = (requested + 1) & 0xFFFFFFFF
            self.sequences[node] = sequence
            reply = dict(origin=request["origin"], destination=node, sequence=sequence, hops=0)
            reverse = self.active(node, request["origin"])
            if reverse:
                self.sim.control("RREP", node, reverse.next_hop, RREP_BODY_BYTES, reply,
                                 self.receive_reply)
        elif hops < request["ttl"]:
            known = self.tables.get(node, {}).get(request["destination"])
            requested = request["destination_sequence"]
            if known and known.sequence_valid and (requested is None
                                                   or sequence_newer(known.sequence, requested)):
                requested = known.sequence
            self.sim.control("RREQ", node, None, RREQ_BODY_BYTES,
                             {**request, "hops": hops, "destination_sequence": requested},
                             self.receive_request)

    def receive_reply(self, node, previous, reply):
        route = self.install(node, reply["destination"], previous, reply["hops"] + 1,
                             reply["sequence"],
                             self.sim.now + self.settings.active_route_timeout)
        if not route.valid or route.sequence != reply["sequence"] or route.next_hop != previous:
            return
        if node != reply["origin"]:
            reverse = self.active(node, reply["origin"])
            if reverse:
                route.precursors.add(reverse.next_hop)
                reverse.precursors.add(previous)
                self.sim.control("RREP", node, reverse.next_hop, RREP_BODY_BYTES,
                                 {**reply, "hops": reply["hops"] + 1}, self.receive_reply)

    def broken_link(self, node, neighbor):
        unreachable = []
        recipients = set()
        for route in self.tables.get(node, {}).values():
            if route.expiry <= self.sim.now:
                route.valid = False
            if route.valid and route.next_hop == neighbor:
                route.valid = False
                route.sequence = (route.sequence + 1) & 0xFFFFFFFF
                route.sequence_valid = True
                unreachable.append((route.destination, route.sequence))
                recipients.update(route.precursors)
        self.send_errors(node, recipients - {neighbor}, unreachable)

    def send_errors(self, node, recipients, unreachable):
        limit = min(255, (self.settings.packet_payload_bytes - 4) // 8)
        reachable = sorted(recipients & set(self.sim.snapshot.adjacency.get(node, {})))
        if not reachable:
            return
        recipient = reachable[0] if len(reachable) == 1 else None
        for offset in range(0, len(unreachable), limit):
            chunk = unreachable[offset:offset + limit]
            self.sim.control("RERR", node, recipient, 4 + 8 * len(chunk), chunk,
                             self.receive_error)

    def receive_error(self, node, previous, unreachable):
        affected = []
        recipients = set()
        for destination, sequence in unreachable:
            route = self.tables.get(node, {}).get(destination)
            if route and route.valid and route.next_hop == previous and (
                    sequence == route.sequence or sequence_newer(sequence, route.sequence)):
                route.valid = False
                route.sequence = sequence
                recipients.update(route.precursors)
                affected.append((destination, sequence))
        self.send_errors(node, recipients - {previous}, affected)

    def path(self, source, destination):
        path = [source]
        while path[-1] != destination:
            node = path[-1]
            route = self.active(node, destination)
            if not route or route.next_hop in path or len(path) > self.settings.network_diameter:
                return ()
            if route.next_hop not in self.sim.snapshot.adjacency.get(node, {}):
                self.broken_link(node, route.next_hop)
                return ()
            path.append(route.next_hop)
        return tuple(path)

    def refresh(self, path):
        for index, node in enumerate(path[:-1]):
            route = self.tables[node][path[-1]]
            route.expiry = self.sim.now + self.settings.active_route_timeout
            if index:
                route.precursors.add(path[index - 1])
        for node in path[1:]:
            route = self.active(node, path[0])
            if route:
                route.expiry = self.sim.now + self.settings.active_route_timeout


# ----------------------------------------------------------------------
# Topology + discrete-event simulator (v1 routing_sim.py, verbatim logic)
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class TopologySnapshot:
    adjacency: object

    def __post_init__(self):
        copied = {node: dict(neighbors) for node, neighbors in self.adjacency.items()}
        for node, neighbors in copied.items():
            for other, distance in neighbors.items():
                if node == other or not math.isfinite(distance) or distance < 0:
                    raise ValueError("invalid topology edge")
                if other not in copied or copied[other].get(node) != distance:
                    raise ValueError("topology must be bidirectional")
        object.__setattr__(self, "adjacency", MappingProxyType({
            node: MappingProxyType(neighbors) for node, neighbors in copied.items()}))

    @classmethod
    def from_edges(cls, nodes, edges):
        adjacency = {node: {} for node in nodes}
        for first, second, distance in edges:
            if first not in adjacency or second not in adjacency:
                raise ValueError("edge refers to unregistered topology node")
            adjacency[first][second] = distance
            adjacency[second][first] = distance
        return cls(adjacency)

    def edges(self):
        return [(node, other, distance) for node in sorted(self.adjacency)
                for other, distance in sorted(self.adjacency[node].items()) if node < other]


@dataclass(frozen=True)
class Delivery:
    message_id: int
    round_num: int
    delivered: bool
    path: tuple
    latency_s: float
    wireless_hops: tuple = ()


class RoutingSimulator:
    def __init__(self, settings=None, capacity: Optional[Callable[[float], float]] = None,
                 seed=None):
        self.settings = settings or AodvSettings()
        if capacity is None:
            raise ValueError("a link-capacity function is required")
        self.capacity = capacity
        self.seed = seed
        self.ledger = RoutingLedger()
        self.now = 0.0
        self.snapshot: TopologySnapshot = TopologySnapshot({})
        self.protocol = AodvProtocol(self, self.settings)
        self._queue: list = []
        self._order = itertools.count()
        self._transmitter_free: Dict[str, float] = {}
        self._message_ids = itertools.count(1)
        self._packet_ids = itertools.count(1)
        self._message_id = 0
        self._round_num = 0
        self._lock = threading.Lock()

    def schedule(self, when, callback, *args):
        heapq.heappush(self._queue, (when, next(self._order), callback, args))

    def drain(self, until=None, deadline=None):
        while self._queue:
            if until is not None and until():
                return
            if deadline is not None and self._queue[0][0] > deadline:
                self.now = max(self.now, deadline)
                return
            when, _, callback, args = heapq.heappop(self._queue)
            self.now = max(self.now, when)
            callback(*args)
        if deadline is not None and not (until is not None and until()):
            self.now = max(self.now, deadline)

    def transmit(self, kind, source, destination, body_bytes, payload, callback,
                 application=0, security=0, packet_id=None):
        neighbors = self.snapshot.adjacency[source]
        recipients = sorted(neighbors) if destination is None else [destination]
        if destination is not None and destination not in neighbors:
            raise ValueError("attempted transmission over absent link")
        distance = max((neighbors[node] for node in recipients), default=0.0)
        capacity = self.capacity(distance)
        if not math.isfinite(capacity) or capacity <= 0:
            raise ValueError("link capacity must be finite and positive")
        start = max(self.now, self._transmitter_free.get(source, 0))
        finish = start + (body_bytes + IP_UDP_HEADER_BYTES) * 8 / capacity
        self._transmitter_free[source] = finish
        if packet_id is None:
            packet_id = next(self._packet_ids)
        event = dict(event="tx", message_id=self._message_id, packet_id=packet_id,
                     round=self._round_num, packet_type=kind, source=source,
                     destination=destination, recipients=recipients, body_bytes=body_bytes,
                     header_bytes=IP_UDP_HEADER_BYTES, capacity_bps=capacity,
                     start_s=start, finish_s=finish)
        if kind == "DATA":
            event.update(fl_application_bytes=application, security_bytes=security)
        else:
            event["control"] = payload
        self.ledger.transmission(event, application, security)
        self.schedule(finish, lambda: None)
        for recipient in recipients:
            arrival = finish + neighbors[recipient] / SPEED_OF_LIGHT + \
                self.settings.node_traversal_time
            self.schedule(arrival, callback, recipient, source, payload)

    def control(self, kind, source, destination, size, payload, callback):
        self.transmit(kind, source, destination, size, payload, callback)

    def submit(self, source, destination, wire_bytes, application_bytes, round_num,
               snapshot, arrival_time=None):
        if not isinstance(snapshot, TopologySnapshot):
            raise ValueError("wireless topology is required")
        if source not in snapshot.adjacency or destination not in snapshot.adjacency \
                or source == destination:
            raise ValueError("invalid wireless endpoints")
        if type(wire_bytes) is not int or type(application_bytes) is not int \
                or not 0 <= application_bytes <= wire_bytes or wire_bytes <= 0:
            raise ValueError("invalid message byte partition")
        if type(round_num) is not int or round_num < 1:
            raise ValueError("round must be a positive integer")
        arrival = float(arrival_time) if arrival_time is not None else 0.0
        if not math.isfinite(arrival) or arrival < 0:
            raise ValueError("invalid arrival time")
        with self._lock:
            self.now = max(self.now, arrival)
            start = self.now
            self._message_id = next(self._message_ids)
            self._round_num = round_num
            self.ledger.submission(round_num, dict(
                event="submission", message_id=self._message_id, round=round_num,
                source=source, destination=destination, wire_bytes=wire_bytes,
                application_bytes=application_bytes, arrival_s=arrival, start_s=start,
                topology_nodes=sorted(snapshot.adjacency),
                topology_edges=snapshot.edges()))
            old_snapshot = self.snapshot
            self.snapshot = snapshot
            for node in sorted(old_snapshot.adjacency):
                for neighbor in sorted(old_snapshot.adjacency[node]):
                    if neighbor not in snapshot.adjacency.get(node, {}):
                        self.ledger.event(dict(event="link_break", message_id=self._message_id,
                                               round=round_num, source=node, neighbor=neighbor,
                                               time_s=self.now))
                        self.protocol.broken_link(node, neighbor)
            self.drain()
            path = self.protocol.path(source, destination)
            if not path:
                self.drain()
                self.protocol.discover(source, destination)
                path = self.protocol.path(source, destination)
            packets = 0
            if path:
                for offset in range(0, wire_bytes, self.settings.packet_payload_bytes):
                    size = min(self.settings.packet_payload_bytes, wire_bytes - offset)
                    application = min(size, max(0, application_bytes - offset))
                    packet_id = next(self._packet_ids)
                    for first, second in zip(path, path[1:]):
                        arrived = []
                        self.transmit("DATA", first, second, size, None,
                                      lambda *args: arrived.append(True),
                                      application, size - application, packet_id)
                        self.drain(until=lambda: bool(arrived))
                        self.protocol.refresh(path)
                    packets += 1
                    self.ledger.event(dict(event="packet_arrival", message_id=self._message_id,
                                           packet_id=packet_id, destination=destination,
                                           round=round_num, time_s=self.now))
            hops = tuple((node, wire_bytes + packets * IP_UDP_HEADER_BYTES,
                          self.capacity(snapshot.adjacency[node][other]))
                         for node, other in zip(path, path[1:]))
            result = Delivery(self._message_id, round_num, bool(path), path,
                              self.now - start, hops)
            self.drain()
            self.ledger.completed(result, packets)
            return result

    def metadata(self, traffic="BC-PAFL FL envelopes", radio_configuration=None,
                 round_arrival_interval_s=90):
        return dict(routing_mode="aodv", model="ideal-link destination-only AODV subset",
                    settings=asdict(self.settings), seed=self.seed, traffic=traffic,
                    radio_configuration=radio_configuration or {},
                    boundary="wireless IPv4/UDP packets; host TCP and wired backhaul excluded",
                    units={"time": "simulated seconds", "volume": "bytes",
                           "NRL": "control TX / final data arrivals"},
                    hello_enabled=False, rrep_ack_enabled=False,
                    routing_control_authenticated=False,
                    application_acceptance="not inferred from network arrival or host "
                                           "handoff; endpoint verification unchanged",
                    assumptions=["immutable topology per message; mobility between messages",
                                 "serialized submissions; ordered trace required for "
                                 "deterministic replay",
                                 "per-transmitter serialization, no contention/fading/MAC "
                                 "ACK/retransmission",
                                 "data fragments forwarded sequentially, no pipelining",
                                 "full-network TTL-limited flood; no intermediate replies "
                                 "or local repair",
                                 "IP/UDP virtual packetization; not host TCP measurement"],
                    control_body_bytes={"RREQ": RREQ_BODY_BYTES, "RREP": RREP_BODY_BYTES,
                                        "RERR": "4 + 8 * destinations"},
                    ip_udp_header_bytes=IP_UDP_HEADER_BYTES,
                    round_arrival_interval_s=round_arrival_interval_s)


# ----------------------------------------------------------------------
# Routing ledger (v1 routing_metrics.py, verbatim logic)
# ----------------------------------------------------------------------
COMPONENTS = ("fl_application_bytes_tx", "security_bytes_tx",
              "routing_control_bytes_tx", "ip_udp_header_bytes_tx")


class RoutingLedger:
    def __init__(self):
        self.events: list = []
        self._rows: Dict[int, dict] = {}
        self._lock = threading.RLock()

    def _row(self, round_num):
        if round_num not in self._rows:
            names = (*COMPONENTS, "total_wireless_bytes_tx", "data_packets_tx",
                     "data_packets_delivered", "messages_submitted", "messages_network_delivered",
                     "messages_no_route", "host_handoffs_succeeded", "host_handoffs_failed",
                     "network_latency_sum_s", "successful_network_latency_sum_s",
                     "rreq_packets_tx", "rrep_packets_tx", "rerr_packets_tx",
                     "rreq_bytes_tx", "rrep_bytes_tx", "rerr_bytes_tx")
            self._rows[round_num] = dict.fromkeys(names, 0)
        return self._rows[round_num]

    def event(self, event):
        with self._lock:
            self.events.append(event)

    def transmission(self, event, application=0, security=0):
        with self._lock:
            row = self._row(event["round"])
            kind = event["packet_type"]
            row["ip_udp_header_bytes_tx"] += IP_UDP_HEADER_BYTES
            row["total_wireless_bytes_tx"] += event["body_bytes"] + IP_UDP_HEADER_BYTES
            if kind == "DATA":
                row["data_packets_tx"] += 1
                row["fl_application_bytes_tx"] += application
                row["security_bytes_tx"] += security
            else:
                row[kind.lower() + "_packets_tx"] += 1
                row[kind.lower() + "_bytes_tx"] += event["body_bytes"] + IP_UDP_HEADER_BYTES
                row["routing_control_bytes_tx"] += event["body_bytes"]
            if sum(row[key] for key in COMPONENTS) != row["total_wireless_bytes_tx"]:
                raise ValueError("wireless byte partition does not conserve volume")
            self.events.append(event)

    def submission(self, round_num, event):
        with self._lock:
            self._row(round_num)["messages_submitted"] += 1
            self.events.append(event)

    def completed(self, delivery, packets):
        with self._lock:
            row = self._row(delivery.round_num)
            row["network_latency_sum_s"] += delivery.latency_s
            row["messages_network_delivered" if delivery.delivered
                else "messages_no_route"] += 1
            if delivery.delivered:
                row["data_packets_delivered"] += packets
                row["successful_network_latency_sum_s"] += delivery.latency_s
            self.events.append(dict(event="network_result", message_id=delivery.message_id,
                                    round=delivery.round_num, delivered=delivery.delivered,
                                    path=delivery.path, latency_s=delivery.latency_s))

    def host_handoff(self, delivery, succeeded):
        with self._lock:
            self._row(delivery.round_num)["host_handoffs_succeeded"
                                          if succeeded else "host_handoffs_failed"] += 1
            self.events.append(dict(event="host_handoff", message_id=delivery.message_id,
                                    round=delivery.round_num, succeeded=bool(succeeded)))

    def rows(self):
        with self._lock:
            result = []
            for round_num, original in sorted(self._rows.items()):
                row = {"round": round_num, **original}
                control = sum(row[k + "_packets_tx"] for k in ("rreq", "rrep", "rerr"))
                row["normalized_routing_load"] = (
                    control / row["data_packets_delivered"]
                    if row["data_packets_delivered"] else math.nan)
                row["network_latency_mean_s"] = (
                    row["network_latency_sum_s"] / row["messages_submitted"])
                row["successful_network_latency_mean_s"] = (
                    row["successful_network_latency_sum_s"] / row["messages_network_delivered"]
                    if row["messages_network_delivered"] else math.nan)
                result.append(row)
            return result

    def export(self, prefix: Path, metadata: dict):
        prefix = Path(prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        with self._lock:
            rows = self.rows()
            with Path(str(prefix) + "_routing_rounds.csv").open(
                    "w", newline="", encoding="utf-8") as stream:
                if rows:
                    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                    writer.writeheader()
                    writer.writerows(rows)
            with Path(str(prefix) + "_routing_events.jsonl").open(
                    "w", encoding="utf-8") as stream:
                for event in self.events:
                    stream.write(json.dumps(event, sort_keys=True) + "\n")
            Path(str(prefix) + "_routing_metadata.json").write_text(
                json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")
