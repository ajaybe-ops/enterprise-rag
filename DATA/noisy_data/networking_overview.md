Networking Technology Overview

A general survey of packet-switched networking. This document covers the OSI
model, circuit switching versus packet switching, and a brief history of the
Internet.

The OSI model is a seven-layer reference architecture. From the bottom up:
Physical, Data Link, Network, Transport, Session, Presentation, Application.
Each layer exposes a service to the layer above and hides its implementation.

Packet switching was first described in a 1968 RAND report. Statistical
multiplexing means that a link carrying variable-length packets can achieve
higher average utilisation than a circuit-switched link with fixed time slots,
because there is no idle capacity reserved for a silent endpoint.

Ethernet defined the framing rules for the first LANs. The 48-bit MAC address
is burned into the interface at manufacture, which is why it is a poor basis
for routing at scale: it is flat, not hierarchical.
