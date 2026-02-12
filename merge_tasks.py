from dedalus.tools import post

file_handlers = ['scalars', 'slices', 'checkpoint', 'final_checkpoint', 'profiles']
# file_handlers = ['scalars', 'slices', 'checkpoint', 'final_checkpoint', 'profiles']
# file_handlers = ['scalars', 'slices', 'checkpoint', 'profiles']
# file_handlers = ['scalars', 'slices', 'profiles']
for task in file_handlers:
    post.merge_analysis(task, cleanup=True)
