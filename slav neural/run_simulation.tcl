# run_simulation.tcl
if {![file exists "work"]} {
    vlib work
}
vmap work work

# Use full path if needed or pass it as argument
vlog full_adder.v
vsim -c top
run -all
exit
