Validation join table notes
===========================

This milestone introduces a lightweight internal ValidationHoleTable used to join
per-hole RDF bootstrap/support outputs with sector propagation and acceleration
validity fractions once, then reuse those aligned arrays across:

- RDF uncertainty table construction
- validation summary JSON writing
- QC gate metric evaluation

The external CSV artifacts remain unchanged. The improvement is internal reuse of
per-hole validation metrics so adjacent validation artifacts do not repeatedly
scan and regroup the same row dictionaries.
