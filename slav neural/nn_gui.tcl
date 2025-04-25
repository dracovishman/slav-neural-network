#!/usr/bin/wish
# nn_gui.tcl - Neural Network Verilog Generator/Analyzer

package require Tk

# Global variables
set verilog_file ""
set perl_script "extract_parameters.pl"
set parameters_file "parameters.txt"
set models_dir "models"

# Create models directory if it doesn't exist
if {![file exists $models_dir]} {
    file mkdir $models_dir
}

# GUI Layout
label .title -text "Neural Network Power Estimation Tool" -font {Arial 16 bold}
pack .title -pady 10

# Control frame with file selection and gate count
frame .controls
button .controls.select -text "1. Select Verilog File" -command select_file
label .controls.gates_label -text "Number of Gates:"
entry .controls.gates_entry -width 10
pack .controls.select .controls.gates_label .controls.gates_entry -side left -padx 5
pack .controls -pady 5

# Action buttons
button .simulate -text "2. Run Simulation" -command run_simulation
pack .simulate -pady 5

button .extract -text "3. Extract Parameters" -command extract_parameters
pack .extract -pady 5

button .train -text "4. Train Power Model" -command run_training
pack .train -pady 5

button .predict -text "5. Predict Power" -command run_prediction
pack .predict -pady 5

# Output display
label .plabel -text "Extracted Parameters / Prediction Output:" -font {Arial 10 bold}
pack .plabel -pady 5

text .text -width 70 -height 15 -wrap word
scrollbar .scroll -command ".text yview"
.text configure -yscrollcommand ".scroll set"
pack .text .scroll -side left -fill y

label .status -text "No file selected yet." -foreground blue
pack .status -pady 10

# Procedures

proc select_file {} {
    global verilog_file
    set verilog_file [tk_getOpenFile -filetypes {{"Verilog Files" {.v}} {"All Files" *}}]
    if {$verilog_file ne ""} {
        .status configure -text "Selected: $verilog_file"
        # Simple gate count estimation based on file size
        set filesize [file size $verilog_file]
        set estimated_gates [expr {int($filesize/500)}]
        .controls.gates_entry delete 0 end
        .controls.gates_entry insert 0 $estimated_gates
    }
}

proc run_simulation {} {
    global verilog_file
    if {$verilog_file eq ""} {
        tk_messageBox -title "Error" -message "Please select a Verilog file first"
        return
    }
    
    set filename [file tail $verilog_file]
    set module_name [file rootname $filename]
    
    set sim_script [open "temp_sim.tcl" w]
    puts $sim_script "vlib work"
    puts $sim_script "vlog \"$verilog_file\""
    puts $sim_script "vsim -voptargs=+acc work.$module_name"
    puts $sim_script "vcd file output.vcd"
    puts $sim_script "vcd add -r /*"
    puts $sim_script "add wave *"
    puts $sim_script "force a 0 0ns, 1 10ns -repeat 20ns"
    puts $sim_script "force b 0 0ns, 1 20ns -repeat 40ns"
    puts $sim_script "force cin 0 0ns, 1 40ns -repeat 80ns"
    puts $sim_script "run 200ns"
    puts $sim_script "quit -sim"
    close $sim_script
    
    .status configure -text "Running simulation..."
    update
    
    if {[catch {exec vsim -c -do temp_sim.tcl} result]} {
        .text insert end "Simulation Error:\n$result\n"
        tk_messageBox -title "Simulation Error" -message "Failed to run simulation"
    } else {
        .text insert end "Simulation completed and VCD generated.\n"
        tk_messageBox -title "Success" -message "Simulation completed successfully"
    }
    
    catch {file delete temp_sim.tcl}
    .status configure -text "Simulation completed"
}

proc extract_parameters {} {
    global perl_script parameters_file
    set num_gates [.controls.gates_entry get]
    
    if {![string is integer $num_gates] || $num_gates <= 0} {
        tk_messageBox -title "Error" -message "Please enter a valid number of gates (positive integer)"
        return
    }
    
    .status configure -text "Extracting parameters..."
    update
    
    if {[catch {exec perl $perl_script $num_gates} result]} {
        .text insert end "Error running Perl script:\n$result\n"
        tk_messageBox -title "Error" -message "Error running Perl script"
    } else {
        if {[file exists $parameters_file]} {
            set f [open $parameters_file r]
            set content [read $f]
            close $f
            .text insert end "Parameters extracted with gate count = $num_gates:\n\n$content\n"
            tk_messageBox -title "Success" -message "Parameters extracted successfully"
        } else {
            .text insert end "Error: parameters.txt not found!\n"
            tk_messageBox -title "Error" -message "parameters.txt not found!"
        }
    }
    .status configure -text "Parameter extraction completed"
}
# Replace the training and prediction procs with these:
proc run_training {} {
    global models_dir
    
    # Create models directory if it doesn't exist
    if {![file exists $models_dir]} {
        file mkdir $models_dir
    }
    
    .status configure -text "Training model (check terminal window)..."
    update
    
    # Run training in a persistent terminal window
    if {$::tcl_platform(platform) eq "windows"} {
        exec cmd.exe /c "start cmd /k python train_model.py" &
    } else {
        exec xterm -hold -e "python train_model.py" &
    }
    
    .status configure -text "Training running in terminal (close when done)"
}

proc run_prediction {} {
    global models_dir
    
    .status configure -text "Running prediction (check terminal window)..."
    .text insert end "\n=== Starting Power Prediction ===\n"
    update
    
    # Check if model files exist
    if {![file exists "$models_dir/scaler.pkl"] || ![file exists "$models_dir/power_model.pth"]} {
        .text insert end "ERROR: Model files not found. Please train the model first.\n"
        .status configure -text "Prediction failed - model not trained"
        return
    }
    
    # Run prediction in a persistent terminal window
    if {$::tcl_platform(platform) eq "windows"} {
        exec cmd.exe /c "start cmd /k python predict_power.py" &
    } else {
        exec xterm -hold -e "python predict_power.py" &
    }
    
    .status configure -text "Prediction running in terminal (close when done)"
    
    # Also capture output for GUI display
    set pipe [open "| python predict_power.py" r]
    fconfigure $pipe -blocking 0
    fileevent $pipe readable [list read_script_output $pipe "predict_power.py"]
}
proc read_script_output {pipe script} {
    if {[eof $pipe]} {
        catch {close $pipe}
        .status configure -text "$script completed"
        .text insert end "\n=== $script finished ===\n"
        return
    }
    
    gets $pipe line
    if {[string length $line] > 0} {
        .text insert end "$line\n"
        .text see end
        update
    }
}