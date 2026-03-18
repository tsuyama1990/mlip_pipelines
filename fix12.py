import re

with open("src/dynamics/dynamics_engine.py", "r") as f:
    text = f.read()

# Replace Template with f-string in _write_cold_start_input
text = re.sub(
    r'        template = string\.Template\("""(.*?)"""\)\n\n        script = template\.substitute\([\s\S]*?work_dir_str=work_dir_str,\n        \)\n',
    r'''        script = f"""units metal
boundary p p p
atom_style atomic

lattice {lattice_type} {float(self.config.lattice_size)}
region box block 0 {int(box_x)} 0 {int(box_y)} 0 {int(box_z)}
create_box 2 box
create_atoms 1 box

# Cold start: using only ZBL
pair_style zbl 1.0 2.0
pair_coeff * * {zbl_mapping}

# Force dump to extract structures for initial training
dump 1 all custom 10 {dump_name} id type x y z
run {int(min(self.config.md_steps, 1000))}
write_restart {work_dir_str}/restart.lammps
write_data {work_dir_str}/data.lammps
"""''',
    text,
    flags=re.DOTALL,
)

# Replace Template with f-string in _write_potential_input
text = re.sub(
    r'        template = string\.Template\("""(.*?)"""\)\n\n        script = template\.substitute\([\s\S]*?work_dir_str=work_dir_str,\n        \)\n',
    r'''        script = f"""units metal
boundary p p p
atom_style atomic

lattice {lattice_type} {float(self.config.lattice_size)}
region box block 0 {int(box_x)} 0 {int(box_y)} 0 {int(box_z)}
create_box 2 box
create_atoms 1 box

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_mapping}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt {self.config.thresholds.smooth_steps} v_max_gamma > {float(self.config.thresholds.threshold_call_dft)} error hard message "AL_HALT"

dump 1 all custom {max(10, self.config.md_steps // 100)} {dump_name} id type x y z c_pace_gamma
run {int(self.config.md_steps)}
write_restart {work_dir_str}/restart.lammps
write_data {work_dir_str}/data.lammps
"""''',
    text,
    count=1,
    flags=re.DOTALL,
)

# Replace Template with f-string in _write_resume_input
text = re.sub(
    r'        template = string\.Template\("""(.*?)"""\)\n\n        script = template\.substitute\([\s\S]*?work_dir_str=str\(work_dir\.resolve\(\)\),\n        \)\n',
    r'''        script = f"""read_restart {str(restart_file.resolve())}

pair_style hybrid/overlay pace zbl 1.0 2.0
pair_coeff * * pace {pot_path_str}
pair_coeff * * zbl {zbl_elements}

compute pace_gamma all pace gamma_mode=1
variable max_gamma equal max(c_pace_gamma)
fix watchdog all halt {self.config.thresholds.smooth_steps} v_max_gamma > {float(self.config.thresholds.threshold_call_dft)} error hard message "AL_HALT"

fix soft_start all langevin {float(self.config.temperature)} {float(self.config.temperature)} 0.1 48279
run 100
unfix soft_start

dump 1 all custom 10 {dump_file_name} id type x y z c_pace_gamma
run {int(self.config.md_steps)}
write_restart {str(work_dir.resolve())}/restart.lammps
write_data {str(work_dir.resolve())}/data.lammps
"""''',
    text,
    flags=re.DOTALL,
)

with open("src/dynamics/dynamics_engine.py", "w") as f:
    f.write(text)
