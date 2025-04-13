resource "openstack_networking_secgroup_v2" "ssh" {
  name        = "ssh-from-all"
  description = "ssh-from-all"
}

resource "openstack_networking_secgroup_rule_v2" "secgroup_ssh_rule_v4" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.ssh.id
}

resource "openstack_networking_secgroup_v2" "openvpn" {
  name        = "openvpn"
  description = "openvpn"
}

resource "openstack_networking_secgroup_rule_v2" "secgroup_openvpn_rule_v4" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 1194
  port_range_max    = 1194
  remote_ip_prefix  = "0.0.0.0/0"
  security_group_id = openstack_networking_secgroup_v2.openvpn.id
}

resource "openstack_networking_secgroup_v2" "ssh-internal" {
  name        = "ssh-internal"
  description = "ssh-internal"
}

resource "openstack_networking_secgroup_rule_v2" "secgroup_ssh-internal_rule_v4" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 22
  port_range_max    = 22
  remote_ip_prefix  = var.network_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.ssh-internal.id
}

resource "openstack_networking_secgroup_v2" "all_internal" {
  name        = "all_internal"
  description = "all_internal"
}

resource "openstack_networking_secgroup_rule_v2" "secgroup_all_internal_rule_tcp_v4" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_ip_prefix  = var.network_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.all_internal.id
}

resource "openstack_networking_secgroup_rule_v2" "secgroup_all_internal_rule_udp_v4" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "udp"
  port_range_min    = 1
  port_range_max    = 65535
  remote_ip_prefix  = var.network_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.all_internal.id
}

resource "openstack_networking_secgroup_v2" "proxy" {
  name        = "proxy"
  description = "proxy"
}

resource "openstack_networking_secgroup_rule_v2" "secgroup_all_internal_rule_http_v4" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 80
  port_range_max    = 80
  remote_ip_prefix  = var.network_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.proxy.id
}

resource "openstack_networking_secgroup_rule_v2" "secgroup_all_internal_rule_https_v4" {
  direction         = "ingress"
  ethertype         = "IPv4"
  protocol          = "tcp"
  port_range_min    = 443
  port_range_max    = 443
  remote_ip_prefix  = var.network_subnet_cidr
  security_group_id = openstack_networking_secgroup_v2.proxy.id
}