# GKE infrastructure for the LLM inference gateway.
# Creates:
#   - GKE cluster with GPU node pools (H100 SXM5 for prefill, A100 PCIe for decode)
#   - VPC with private subnets
#   - Cloud Filestore PVC for model weights
#   - Redis (Memorystore) for distributed prefix cache metadata

terraform {
  required_version = ">= 1.7"
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    google-beta = {
      source  = "hashicorp/google-beta"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.25"
    }
  }
  backend "gcs" {
    bucket = "YOUR_TERRAFORM_STATE_BUCKET"
    prefix = "inference-gateway/terraform"
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

provider "google-beta" {
  project = var.project_id
  region  = var.region
}

# ── Variables ─────────────────────────────────────────────────────────────

variable "project_id" {
  description = "GCP project ID"
  type        = string
}

variable "region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "GCP zone for zonal resources"
  type        = string
  default     = "us-central1-a"
}

variable "cluster_name" {
  description = "GKE cluster name"
  type        = string
  default     = "inference-gateway"
}

variable "prefill_node_count" {
  description = "Number of H100 prefill nodes"
  type        = number
  default     = 8
}

variable "decode_node_count" {
  description = "Number of A100 decode nodes"
  type        = number
  default     = 16
}

# ── VPC ───────────────────────────────────────────────────────────────────

resource "google_compute_network" "inference_vpc" {
  name                    = "${var.cluster_name}-vpc"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "inference_subnet" {
  name          = "${var.cluster_name}-subnet"
  ip_cidr_range = "10.0.0.0/16"
  region        = var.region
  network       = google_compute_network.inference_vpc.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }
  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/16"
  }
}

# ── GKE Cluster ───────────────────────────────────────────────────────────

resource "google_container_cluster" "inference_cluster" {
  provider = google-beta
  name     = var.cluster_name
  location = var.zone

  # Use a separately-managed node pool.
  remove_default_node_pool = true
  initial_node_count       = 1

  network    = google_compute_network.inference_vpc.name
  subnetwork = google_compute_subnetwork.inference_subnet.name

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  private_cluster_config {
    enable_private_nodes    = true
    enable_private_endpoint = false
    master_ipv4_cidr_block  = "172.16.0.0/28"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }

  addons_config {
    gce_persistent_disk_csi_driver_config { enabled = true }
    gcs_fuse_csi_driver_config { enabled = true }
    ray_operator_config {
      enabled            = true
      ray_cluster_logging_config { enabled = true }
      ray_cluster_monitoring_config { enabled = true }
    }
  }

  release_channel {
    channel = "RAPID"
  }

  maintenance_policy {
    recurring_window {
      start_time = "2024-01-01T02:00:00Z"
      end_time   = "2024-01-01T06:00:00Z"
      recurrence = "FREQ=WEEKLY;BYDAY=SA,SU"
    }
  }
}

# ── Head/CPU node pool ────────────────────────────────────────────────────

resource "google_container_node_pool" "cpu_pool" {
  name     = "cpu-pool"
  cluster  = google_container_cluster.inference_cluster.name
  location = var.zone

  initial_node_count = 3
  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }

  node_config {
    machine_type = "n2-standard-32"
    disk_type    = "pd-ssd"
    disk_size_gb = 200
    labels = { "node-role" = "inference-head" }
    workload_metadata_config { mode = "GKE_METADATA" }
    shielded_instance_config {
      enable_secure_boot          = true
      enable_integrity_monitoring = true
    }
  }
}

# ── Prefill GPU node pool (H100 SXM5) ────────────────────────────────────

resource "google_container_node_pool" "prefill_pool" {
  provider = google-beta
  name     = "prefill-pool"
  cluster  = google_container_cluster.inference_cluster.name
  location = var.zone

  initial_node_count = var.prefill_node_count
  autoscaling {
    min_node_count = 4
    max_node_count = 32
  }

  node_config {
    machine_type = "a3-highgpu-8g"  # 8x H100 SXM5 80GB
    disk_type    = "pd-ssd"
    disk_size_gb = 1000

    guest_accelerator {
      type  = "nvidia-h100-80gb"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    labels = {
      "node-role"                           = "prefill"
      "cloud.google.com/gke-accelerator"   = "nvidia-h100-80gb"
    }
    taints = [
      { key = "prefill-only", value = "true", effect = "NO_SCHEDULE" },
      { key = "nvidia.com/gpu", value = "present", effect = "NO_SCHEDULE" },
    ]

    reservation_affinity {
      consume_reservation_type = "SPECIFIC_RESERVATION"
      key                      = "compute.googleapis.com/reservation-name"
      values                   = ["h100-reservation"]
    }

    workload_metadata_config { mode = "GKE_METADATA" }
  }
}

# ── Decode GPU node pool (A100 PCIe) ──────────────────────────────────────

resource "google_container_node_pool" "decode_pool" {
  name     = "decode-pool"
  cluster  = google_container_cluster.inference_cluster.name
  location = var.zone

  initial_node_count = var.decode_node_count
  autoscaling {
    min_node_count = 8
    max_node_count = 64
  }

  node_config {
    machine_type = "a2-highgpu-8g"  # 8x A100 PCIe 80GB
    disk_type    = "pd-ssd"
    disk_size_gb = 500

    guest_accelerator {
      type  = "nvidia-a100-80gb"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    labels = {
      "node-role"                         = "decode"
      "cloud.google.com/gke-accelerator" = "nvidia-a100-80gb"
    }
    taints = [
      { key = "decode-only", value = "true", effect = "NO_SCHEDULE" },
      { key = "nvidia.com/gpu", value = "present", effect = "NO_SCHEDULE" },
    ]

    workload_metadata_config { mode = "GKE_METADATA" }
  }
}

# ── Spot decode pool (cost-optimised for batch workloads) ─────────────────

resource "google_container_node_pool" "decode_spot_pool" {
  name     = "decode-spot-pool"
  cluster  = google_container_cluster.inference_cluster.name
  location = var.zone

  autoscaling {
    min_node_count = 0
    max_node_count = 32
  }

  node_config {
    machine_type = "a2-highgpu-8g"
    spot         = true   # Up to 70% cheaper; may be preempted
    disk_type    = "pd-ssd"
    disk_size_gb = 500

    guest_accelerator {
      type  = "nvidia-a100-80gb"
      count = 8
      gpu_driver_installation_config {
        gpu_driver_version = "LATEST"
      }
    }

    labels = {
      "node-role"                         = "decode"
      "workload-type"                     = "spot"
      "cloud.google.com/gke-accelerator" = "nvidia-a100-80gb"
    }
    taints = [
      { key = "decode-only", value = "true", effect = "NO_SCHEDULE" },
      { key = "nvidia.com/gpu", value = "present", effect = "NO_SCHEDULE" },
      { key = "cloud.google.com/gke-spot", value = "true", effect = "NO_SCHEDULE" },
    ]

    workload_metadata_config { mode = "GKE_METADATA" }
  }
}

# ── Cloud Filestore (model weights NFS) ───────────────────────────────────

resource "google_filestore_instance" "model_weights" {
  name     = "model-weights"
  location = var.zone
  tier     = "ENTERPRISE"

  file_shares {
    name        = "model_cache"
    capacity_gb = 10240  # 10 TB for large model weights
  }

  networks {
    network = google_compute_network.inference_vpc.name
    modes   = ["MODE_IPV4"]
  }
}

# ── Redis Memorystore (prefix cache metadata) ─────────────────────────────

resource "google_redis_cluster" "prefix_cache" {
  provider       = google-beta
  name           = "prefix-cache"
  region         = var.region
  shard_count    = 3
  replica_count  = 1
  node_type      = "REDIS_HIGHMEM_MEDIUM"
  transit_encryption_mode = "TRANSIT_ENCRYPTION_MODE_SERVER_AUTHENTICATION"
  authorization_mode      = "AUTH_MODE_IAM_AUTH"

  psc_configs {
    network = google_compute_network.inference_vpc.id
  }
}

# ── Outputs ───────────────────────────────────────────────────────────────

output "cluster_name" {
  value = google_container_cluster.inference_cluster.name
}

output "cluster_endpoint" {
  value     = google_container_cluster.inference_cluster.endpoint
  sensitive = true
}

output "redis_host" {
  value     = google_redis_cluster.prefix_cache.discovery_endpoints[0].address
  sensitive = true
}

output "filestore_ip" {
  value = google_filestore_instance.model_weights.networks[0].ip_addresses[0]
}
