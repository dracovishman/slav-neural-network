#!/usr/bin/perl
use strict;
use warnings;

my $num_gates = $ARGV[0] || 1000;  
my $vcd_file = "output.vcd";
open my $fh, '<', $vcd_file or die "Cannot open $vcd_file: $!";

my ($num_transitions, $last_time, $has_transitions) = (0, 0, 0);

while (<$fh>) {
    if (/^#(\d+)/) {
        $last_time = $1;
    }
    if (/^[01xz]/) {
        $num_transitions++;
        $has_transitions = 1;
    }
}

close $fh;

# Calculate and bound switching activity
my $switching_activity = 0.25; # default
if ($has_transitions && $last_time > 0) {
    $switching_activity = ($num_transitions/$num_gates) / ($last_time/1e9) / 1e6;
    $switching_activity = 0.1 if $switching_activity < 0.1;
    $switching_activity = 0.5 if $switching_activity > 0.5;
}

open my $out_fh, '>', 'parameters.txt' or die "Cannot write to parameters.txt: $!";
print $out_fh "avg_gate_cap 0.1\n";
print $out_fh "clock_factor 1.0\n";
print $out_fh "tech_node 28\n";
print $out_fh "vdd 1.2\n";
print $out_fh "frequency 2\n";
print $out_fh "switching_activity $switching_activity\n";
print $out_fh "num_gates $num_gates\n";
print $out_fh "temperature 25\n";
close $out_fh;

print "Parameters written to parameters.txt\n";