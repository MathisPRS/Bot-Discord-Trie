variable "network_external_id" {
  type    = string
  default = "34a684b8-2889-4950-b08e-c33b3954a307"
}

variable "network_external_name" {
  type    = string
  default = "ext-floating1"
}

variable "network_internal_dev" {
  type    = string
  default = "internal_dev"
}

variable "network_subnet_cidr" {
  type    = string
  default = "10.0.1.0/24"
}

variable "ssh_public_key_default_user" {
  type    = string
  default = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIC/5C9DTsb81GFOqtScetpXs1ENr9SWCTcv2TuXgo9lz mathis@DESKTOP-IR84FN6"
}

variable "instance_image_id" {
  type    = string
  default = "a0c44881-50cc-4c9d-9753-89206cff776d"
}

variable "instance_flavor_name" {
  type    = string
  default = "a8-ram32-disk50-perf1"
}

variable "instance_security_groups" {
  type    = list(any)
  default = ["default"]
}

variable "metadatas" {
  type = map(string)
  default = {
    "environment" = "dev"
  }
}