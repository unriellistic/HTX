"""
Just a mini testing file for me to try out python logic for debugging.
"""
total_area_of_annotation_cut = 0.01
total_object_annotation_area = 500

plane_has_info_loss_questionmark = {"plane 1": [False, 100]}

left=(plane_has_info_loss_questionmark["plane 1"][1] if plane_has_info_loss_questionmark["plane 1"][0]==False else 1000000)
print(left)