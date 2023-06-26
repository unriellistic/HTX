import general_scripts as gs

dualenergy_files = gs.load_files(r"H:\busxray\master_file_for_both_clean_and_threat_images_dualenergy")
monochrome_files = gs.load_files(r"H:\busxray\master_file_for_both_clean_and_threat_images_monochrome")
dualenergy_files_xml = gs.load_files(r"H:\busxray\master_file_for_both_clean_and_threat_images_dualenergy", file_type=".xml")
monochrome_files_xml = gs.load_files(r"H:\busxray\master_file_for_both_clean_and_threat_images_monochrome", file_type=".xml")

clean_both = gs.load_files(r"H:\busxray\Compiled_Clean_Images")
threat_both = gs.load_files(r"H:\busxray\Compiled_Threat_Images\removeThreat_images")
clean_both_xml = gs.load_files(r"H:\busxray\Compiled_Clean_Images", file_type=".xml")
threat_both_xml = gs.load_files(r"H:\busxray\Compiled_Threat_Images\removeThreat_images", file_type=".xml")

clean_both_de = [gs.path_leaf(i)[1] for i in clean_both if "Dual" in i]
clean_both_mc = [gs.path_leaf(i)[1] for i in clean_both if "Mono" in i]
threat_both_de = [gs.path_leaf(i)[1] for i in threat_both if "final" in i]
threat_both_mc = [gs.path_leaf(i)[1] for i in threat_both if "temp" in i]

gs.print_nice_lines("=", 1)
print("dualenergy_files:", len(dualenergy_files))
print("monochrome_files:", len(monochrome_files))

print("clean_both:", len(clean_both))
print("--> clean_both_de:", len(clean_both_de))
print("--> clean_both_mc:", len(clean_both_mc))
print("threat_both:", len(threat_both))
print("--> threat_both_de:", len(threat_both_de))
print("--> threat_both_mc:", len(threat_both_mc))
gs.print_nice_lines("-", 1)
# 2 images got the yellow streak across, taken out of dataset 4-7
if len(dualenergy_files) == len(clean_both_de)+len(threat_both_de)-2:
    print("de file check pass...")
else:
    print("de check failed...")
if len(monochrome_files) == len(clean_both_mc)+len(threat_both_mc)-2:
    print("mc file check pass...")
else:
    print(f"mc check failed. Shortage of {len(monochrome_files) - len(clean_both_mc) - len(threat_both_mc)-2}")
gs.print_nice_lines("=", 1)



print("dualenergy_files_xml:", len(dualenergy_files_xml))
print("monochrome_files_xml:", len(monochrome_files_xml))
print("clean_both_xml:", len(clean_both_xml))
print("threat_both_xml:", len(threat_both_xml))



print("-"*20)

count = 0
for f in monochrome_files:
    _, fname = gs.path_leaf(f)
    if fname not in clean_both_mc or fname not in threat_both_mc:
        # print(fname)
        count +=1
print("count:", count)