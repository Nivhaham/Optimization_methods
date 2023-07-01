from scapy.all import *
a = Ether() / IP(dst="8.8.8.8", ttl=64) / ICMP()

srp1(a)
