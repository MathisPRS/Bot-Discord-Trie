resource "openstack_networking_router_v2" "rt1" {
  name                = "rt1"
  admin_state_up      = "true"
  external_network_id = var.network_external_id
}

resource "openstack_networking_network_v2" "network_internal" {
  name           = var.network_internal_dev
  admin_state_up = "true"
}

resource "openstack_networking_subnet_v2" "network_subnet" {
  name            = var.network_internal_dev
  network_id      = openstack_networking_network_v2.network_internal.id
  cidr            = var.network_subnet_cidr
  ip_version      = 4
  enable_dhcp     = true
  dns_nameservers = ["1.1.1.1", "8.8.8.8"]
}

resource "openstack_networking_router_interface_v2" "network_router_interface" {
  router_id = openstack_networking_router_v2.rt1.id
  subnet_id = openstack_networking_subnet_v2.network_subnet.id
}

