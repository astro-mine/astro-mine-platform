{{/* Common name + label helpers for the Astro-Mine-Cloud platform chart. */}}

{{- define "astro-mine-cloud.name" -}}
astro-mine-cloud
{{- end -}}

{{- define "astro-mine-cloud.namespace" -}}
{{- .Values.platform.namespace | default "astro-mine-system" -}}
{{- end -}}

{{/* The standard label set, mirroring astro_mine.cloud.k8s.labels(). */}}
{{- define "astro-mine-cloud.labels" -}}
app.kubernetes.io/name: {{ include "astro-mine-cloud.name" . }}
app.kubernetes.io/managed-by: astro-mine-cloud
app.kubernetes.io/part-of: astro-mine
helm.sh/chart: astro-mine-cloud-{{ .Chart.Version }}
{{- end -}}
