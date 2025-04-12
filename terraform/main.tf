# Define required providers
terraform {
  required_providers {
    openstack = {
      source  = "terraform-provider-openstack/openstack"
      version = "1.44.0"
    }
  }
}

# Configure the OpenStack Provider
provider "openstack" {
}


resource "openstack_compute_keypair_v2" "keypair_mathis" {
  name = "keypair_mathis"
  public_key = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC/5C9DTsb81GFOqtScetpXs1ENr9SWCTcv2TuXgo9lz mathis@DESKTOP-IR84FN6"
}


# Create a web security group
resource "openstack_compute_secgroup_v2" "web_server" {
  name        = "web_server"
  description = "Security Group Description"

  rule {
    from_port   = 22
    to_port     = 22
    ip_protocol = "tcp"
    cidr        = "0.0.0.0/0"
  }

  rule {
    from_port   = 80
    to_port     = 80
    ip_protocol = "tcp"
    cidr        = "0.0.0.0/0"
  }

  rule {
    from_port   = 443
    to_port     = 443
    ip_protocol = "tcp"
    cidr        = "0.0.0.0/0"
  }
}


# Create a web server
resource "openstack_compute_instance_v2" "web-server" {
  name            = "web-server"
  image_id        = "a0c44881-50cc-4c9d-9753-89206cff776d"
  flavor_name     = "a2-ram4-disk50-perf1"
  key_pair        = "keypair_mathis"
  security_groups = ["web_server"]

  metadata = {
    application = "web-app"
  }

  network {
    name = "ext-net1"
  }
}
