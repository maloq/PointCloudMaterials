"""Predeclared one-factor radius, optimization and data scaling comparisons."""


def variants(settings):
    result=[]
    def add(mode,tensor,radius=25,epochs=3,sources=90,fraction=1.,aggregation='attention',budget='epochs'):
        name=f'{mode}'+('-tensor' if tensor else '')+f'-H12-R{radius}-{aggregation}-E{epochs}-S{sources}-W{fraction:g}'
        if budget=='full_data_epochs':name+='-matched-updates'
        if any(s['name']==name for s in result):return
        result.append(dict(name=name,mode=mode,history_ps=12,radius_A=radius,aggregation=aggregation,
            equivariant=tensor,baseline=None,repeat=False,
            training=dict(budget=budget,epochs=epochs,sources=sources,window_fraction=fraction)))
    models=[('frozen',False),('finetune',False),('scratch',False),('frozen',True),('finetune',True)]
    # Useful short checkpoints first; six-epoch fits follow the cheaper factors.
    for epochs in settings['epochs'][:-1]:
        for mode,tensor in models:add(mode,tensor,epochs=epochs)
    for radius in settings['radii_A']:
        for mode,tensor in models:add(mode,tensor,radius=radius)
    for sources in settings['source_counts']:
        if sources==90:continue  # Full-data E3 is the common reference.
        for mode,tensor in models:add(mode,tensor,sources=sources,budget='full_data_epochs')
    for fraction in settings['window_fractions']:
        if fraction==1:continue
        for mode in ('frozen','finetune','scratch'):add(mode,False,fraction=fraction,budget='full_data_epochs')
    for radius in settings['radii_A']:
        if radius>0:add('frozen',False,radius=radius,aggregation='mean')
    for mode,tensor in models:add(mode,tensor,epochs=settings['epochs'][-1])
    return result
