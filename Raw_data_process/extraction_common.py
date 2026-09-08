"""Portable raw-data parsing and extraction; previous numerical results are never inputs."""
from pathlib import Path
from datetime import time as ExcelTime
from collections import defaultdict, Counter
import csv
import numpy as np

import openpyxl
RAW = Path(__file__).resolve().parent
ROOT = RAW.parent
import sys
sys.dont_write_bytecode = True
from output_schema import SCHEMAS
OUT = ROOT / "data"
OUT.mkdir(exist_ok=True)
G = ['Ai','Ac','Bfi','Bf','Bt','Bu','Bx','Ba','Csp','Dl','Ls','Mf','Pm','Bd','Bv','Rg']
NG = ['Af','Ao','As','Bl.s','Col','Cs','Et','Eu.c','Eu.l','Im','Ld','Lsp','Mi','Pc','Va','Vp']
ALL = G + NG
FORTY = G + ['Bb','Bbi','Bc','Bl','Bt47','Bt61','Pg','Pj'] + NG
CORE13 = ['Ai','Bt','Csp','Dl','Mf','Bd','Cs','Eu.c','Eu.l','Im','Mi','Va','Vp']
ALIASES = {'Lsp50':'Lsp','Afi':'Af','Bls':'Bl.s','Euc':'Eu.c','Eul':'Eu.l'}

def parse_float_list(value):
    return np.fromstring(str(value), sep=" ").tolist()

def parse_time_axis(value, n):
    if isinstance(value, ExcelTime):
        # The supplied Ld range cells were autoformatted as clock times in Excel.
        return np.linspace(0.0, 48.0, n).tolist()
    parts = str(value).replace(" ", "").split(":")
    if len(parts) == 3:
        return np.linspace(float(parts[0]), float(parts[2]), n).tolist()
    return parse_float_list(value)

def text(x):
    return '' if x is None else str(x).strip()


def num(x):
    try:
        v=float(x)
        return v if np.isfinite(v) else None
    except (ValueError,TypeError):
        return None


def read_excel(relative):
    p=RAW/relative
    book=openpyxl.load_workbook(p,read_only=True,data_only=True)
    rows=[]
    for sheet in book:
        iterator=iter(sheet.values)
        headers=[text(v) for v in next(iterator)]
        species_headers=headers[headers.index('Ai'):headers.index('Vp')+1] if 'Ai' in headers and 'Vp' in headers else []
        for i,values in enumerate(iterator,2):
            if not any(v is not None for v in values):continue
            row={ALIASES.get(k,k):v for k,v in zip(headers,values) if k}
            row.update(source_file=str(p.relative_to(ROOT)),source_sheet=sheet.title,source_excel_row=i,
                       source_key=f'{relative}|{sheet.title}|{i}',_species_headers=species_headers)
            rows.append(row)
    return rows


def meta(r):
    return {k:r.get(k,'') for k in ['source_file','source_sheet','source_excel_row','source_key']} | {
        'sample_id':r.get('Sample ID',r['source_key']),
        'experiment_id':r.get('Experiment ID',r.get('ExperimentID','')),
        'growth_well_id':r.get('Growth Well ID',''),
        'raw_group':r.get('Group',r.get('Species','')),
        'source_note':text(r.get('Note')),
        'time_axis_note':'Excel clock-time cell: assumed agreed 0-48 h protocol' if isinstance(r.get('Growth curve time'), ExcelTime) else ''}


def curve(r):
    raw=np.array(parse_float_list(r.get('Growth curve OD',r.get('Growth curve'))),float)
    times=np.array(parse_time_axis(r.get('Growth curve time'),len(raw)),float)
    if len(raw)<2 or len(raw)!=len(times) or not np.isfinite(raw).all() or not np.isfinite(times).all():
        raise ValueError(f'Invalid curve: {r["source_key"]}')
    return times,raw


def od_metrics(r):
    t,y=curve(r)
    return dict(initial_od=float(y[0]),final_od=float(y[-1]),minimum_od=float(y.min()),
                final_minus_initial=float(y[-1]-y[0]),final_minus_minimum=float(y[-1]-y.min()))


def exclusion_note(r):
    note=text(r.get('Note')).lower().replace(' ','_')
    if 'contamin' in note:return 'source_contamination'
    if 'not_used' in note:return 'source_not_used'
    return ''


def save_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    if fields is None:fields=list(dict.fromkeys(k for r in rows for k in r))
    def cv(v):
        return '' if v is None or isinstance(v,(float,np.floating)) and not np.isfinite(v) else v
    with path.open('w',newline='') as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction='ignore');w.writeheader()
        w.writerows({k:cv(r.get(k)) for k in fields} for r in rows)


def schema(filename):
    return SCHEMAS[filename]["columns"]


def complete(filename,rows,replicates,sources,method,fields=None,extra=None):
    definition = SCHEMAS[filename]
    key = definition["row_key"]
    by_key = {str(row[key]): row for row in rows}
    assert len(by_key) == len(rows), f"Duplicate row labels in {filename}"
    missing = set(definition["rows"]) - set(by_key)
    assert not missing, f"Missing output rows in {filename}: {missing}"
    # Preserve the published layout; extra excluded conditions remain in the source workbook.
    rows = [by_key[label] for label in definition["rows"]]
    save_csv(OUT/filename, rows, definition["columns"])
    print(f'{filename}: {len(rows)} rows saved', flush=True)



def matrix(row_species,columns,values,index='Species'):
    return [{index:s,**{c:values.get((s,c)) for c in columns}} for s in row_species]


def growth_matrix(filename,sources,row_species,columns,kind):
    audit=[];group=defaultdict(list);alt=defaultdict(list)
    for source in sources:
        for r in read_excel(source):
            protocol=text(r.get('Species',r.get('Group')))
            parts=protocol.split('_')
            if kind=='double' and 'Non_growers_' in source:
                if len(parts)!=3 or parts[0]!='Bt' or parts[2] not in ('100%','10%'):continue
                c='Full_Bt' if parts[2]=='100%' else 'Diluted_Bt';s=parts[1];metric='final_minus_initial'
            elif kind=='double':
                if len(parts)!=2:continue
                c,s=parts;metric='final_minus_minimum'
            elif kind=='nongrower_spent':
                if len(parts)!=3 or parts[2]!='100%':continue
                c,s=parts[:2];metric='final_minus_initial'
            else:
                if len(parts)!=2:continue
                c,s=parts;metric='final_minus_initial'
            s=ALIASES.get(s,s);c=ALIASES.get(c,c)
            if s not in row_species or c not in columns:continue
            m=od_metrics(r);reason=exclusion_note(r)
            item=meta(r)|m|dict(focal_species=s,conditioner=c,metric=metric,excluded_reason=reason,included=not bool(reason))
            audit.append(item)
            if not reason:group[s,c].append(m[metric]);alt[s,c].append(m['final_minus_initial'])
    values={};cell_audit=[]
    for s in row_species:
        for c in columns:
            v=group.get((s,c),[])
            value=None
            if v:
                if kind=='grower_spent':
                    # Historical strict trigger; decimal tolerance prevents roundoff promoting exactly .01.
                    trigger=any(x>.01+1e-12 for x in v)
                    value=float(np.maximum(v,0).mean()) if trigger else 0.
                else:value=max(float(np.mean(v)),0.)
            values[s,c]=value
            cell_audit.append(dict(focal_species=s,conditioner=c,n_replicates=len(v),processed_growth=value,
                raw_mean=float(np.mean(v)) if v else None,raw_sd=float(np.std(v,ddof=1)) if len(v)>1 else None,
                mean_final_minus_initial=float(np.mean(alt[s,c])) if v else None))
    idx='focal_species' if kind=='grower_spent' else 'Species'
    method={'double':'Double-spent cells: mean(final minus per-curve minimum); Full_Bt/Diluted_Bt: max(mean(final minus initial),0).',
            'nongrower_spent':'100% only; max(mean(final minus initial),0); every unexcluded replicate equally weighted.',
            'grower_spent':'If any unexcluded replicate has final-minus-initial >0.01, mean(max(replicate delta,0)); otherwise zero. Low_growth note alone does not remove repeats: condition-level trigger is used.'}[kind]
    complete(filename,matrix(row_species,columns,values,idx),audit,sources,method,
             extra={'missing_cells':sum(v is None for v in values.values())})


def designed(group,kind):
    g=text(group)
    if kind=='ng':
        if g in ('Full_community','Bt'):return NG.copy(),'Full_community'
        if g.endswith('_dropout'):
            sp=ALIASES.get(g[:-8],g[:-8]);return [s for s in NG if s!=sp],sp+'_dropout'
        pool=NG
    else:
        if g=='Full_community':return ALL.copy(),'Full_community_32_Glucose'
        if g=='Whole_community':return CORE13.copy(),'Whole_community_13'
        if g in ('Bt_dropout','Bd_dropout','Bt_Bd_dropout'):
            drops=g.removesuffix('_dropout').split('_');return [s for s in CORE13 if s not in drops],g
        pool=ALL
    species=[ALIASES.get(s,s) for s in g.split('_')]
    if not all(s in pool for s in species):raise ValueError(f'Unknown community {g}')
    return sorted(set(species),key=pool.index),None


def adjusted_counts(r,species):
    counts={s:num(r.get(s)) or 0. for s in species}
    # Preserve two presented channels when both designed; recover shared Eu.c signal when only Eu.l designed.
    if 'Eu.l' in species and 'Eu.c' not in species:
        counts['Eu.l']+=num(r.get('Eu.c')) or 0.
    return counts


def community_matrix(filename,source,kind,*,user_excluded_communities=(),user_excluded_samples=()):
    pool=NG if kind=='ng' else ALL
    # Old header controls labels/order only; membership is inferred from experimental protocol.
    oldcols=schema(filename)[1:];name_by_set={}
    for col in oldcols:
        if 'community' not in col.lower() and 'dropout' not in col.lower():
            p=[ALIASES.get(s,s) for s in col.split('_')]
            if all(s in pool for s in p):name_by_set[frozenset(p)]=col
    audit=[];accepted=defaultdict(list);members={}
    for r in read_excel(source):
        species,label=designed(r.get('Group'),kind)
        name=label or name_by_set.get(frozenset(species),'_'.join(species));members[name]=species
        counts=adjusted_counts(r,species);den=sum(counts.values())
        all_reads=sum(num(r.get(s)) or 0 for s in r['_species_headers'])
        off=(all_reads-den)/all_reads if all_reads>0 else None
        sample_key=(text(r.get('Experiment ID')),text(r.get('Growth Well ID')))
        user_reason='user_excluded_community' if name in user_excluded_communities else (
            'user_excluded_sample' if sample_key in user_excluded_samples else '')
        reason=exclusion_note(r)
        if not reason and 'low_growth' in text(r.get('Note')).lower():reason='source_low_growth'
        if not reason and user_reason:reason=user_reason
        if not reason and den<=0:reason='no_designed_species_reads'
        # Same high off-target exclusion used by the repository's simple community-QC branches.
        if not reason and off is not None and off>.10:reason='off_design_fraction_gt_0.10'
        rel={s:counts[s]/den for s in species} if den>0 else {}
        qc='contaminated' if 'contamin' in reason else 'suspect' if reason else 'clean'
        item=meta(r)|dict(community=name,designed_species=';'.join(species),designed_reads=den,
            total_reads=all_reads,off_design_fraction=off,included=not bool(reason),excluded_reason=reason,
            user_exclusion_reason=user_reason,
            contamination_status=qc,modeling_action_default='exclude' if reason else 'allow',
            relative_sum=sum(rel.values()) if rel else None)
        item.update({s:rel.get(s) for s in pool});audit.append(item)
        if not reason:accepted[name].append(rel)
    columns=[c for c in oldcols if c in members]+sorted(set(members)-set(oldcols))
    values={};stats=[]
    for c in columns:
        group=accepted[c]
        for s in members[c]:values[s,c]=float(np.mean([r[s] for r in group])) if group else None
        stats.append(dict(community=c,n_used=len(group),n_total=sum(r['community']==c for r in audit),
            sum_relative=sum(values[s,c] for s in members[c]) if group else None))
    for r in stats:
        if r['n_used']:assert np.isclose(r['sum_relative'],1.)
    complete(filename,matrix(pool,columns,values,'species'),audit,[source],
        'Normalize each raw replicate over designed species (shared Eu readout handled within replicate), then arithmetic mean of replicate relative vectors. Exclude source Contamination/Not_used/Low_growth, explicit user-excluded communities/samples, zero designed reads, off-design fraction >0.10. Absent species remain NA; all-excluded communities remain NA.',
        extra={'community_count':len(columns),'retained_replicates':sum(r['included'] for r in audit),
               'excluded_reason_counts':dict(Counter(r['excluded_reason'] for r in audit if not r['included'])),
               'user_excluded_communities':list(user_excluded_communities),
               'user_excluded_samples':[dict(experiment_id=batch,growth_well_id=well) for batch,well in user_excluded_samples],
               'all_excluded_communities':[r['community'] for r in stats if not r['n_used']]})


def grower_dm(filename,source):
    audit=[];group=defaultdict(list)
    for r in read_excel(source):
        s=ALIASES.get(text(r['Species']),text(r['Species']))
        if s not in G or text(r['Media'])!='DM68':continue
        reason=exclusion_note(r);m=od_metrics(r)
        audit.append(meta(r)|m|dict(species=s,included=not bool(reason),excluded_reason=reason))
        if not reason:group[s].append(m['final_minus_initial'])
    result=[dict(species=s,mean_growth=float(np.mean(group[s])) if group[s] else None) for s in G]
    complete(filename,result,audit,[source],'Mean raw final-minus-initial over all unexcluded DM68 monoculture replicates. No clipping or growth-trigger selection.')


def carbon_absolute(filename,source):
    output=[];audit=[];relative=[];by_media=defaultdict(list)
    headers=schema(filename)
    for r in read_excel(source):
        counts={s:num(r.get(s)) or 0. for s in FORTY};den=sum(counts.values())
        canonical=sum(counts[s] for s in ALL)
        final=num(r.get('Final OD'));assert final is not None and den>0
        rel={s:counts[s]/den for s in FORTY};absolute={s:final*rel[s] for s in FORTY}
        note=exclusion_note(r)
        raw_qc=text(r.get('Note')).lower().replace('_', ' ')
        # User-requested legacy output label; preserve the actual raw reason in the source workbook.
        qc='high other species' if raw_qc == 'not used' else raw_qc
        item={'Sample ID':r.get('Sample ID'),'Experiment ID':r.get('Experiment ID'),'growth_qc':qc,
            'Growth Well ID':r.get('Growth Well ID'),'16S Well ID':r.get('16S Well ID'),'Media':r.get('Media'),
            'Protocol':r.get('Protocol'),'Initial OD':'','Final OD':final,'Growth curve time':r.get('Growth curve time'),
            'Growth curve OD':r.get('Growth curve OD'),'Notes':r.get('Note'),'Data location':r['source_file']}
        # Reproduce the published metadata convention; raw identities remain in the source workbook.
        plate, well = text(r['Growth Well ID']).split('_', 1)
        batch = f'2024930_WZY_DM_assembly_{plate}'
        item.update({
            'Sample ID': batch + well,
            'Experiment ID': batch,
            'Growth Well ID': well,
            'Notes': 'growth curve data NA, so 48 h label rather than 0: 0.05:48 was used',
            'Data location': f'Ho lab/7_Shared data/16s rRNA/Raw reads/2024930_DM_assembly/{plate}',
        })
        item.update(absolute);output.append(item)
        include=not qc and text(r['Media']).lower()!='blank' and canonical>0
        audit.append(meta(r)|dict(media=r['Media'],final_od=final,selected40_reads=den,canonical32_reads=canonical,
            growth_qc=qc,raw_growth_qc=raw_qc,included_in_mean_relative=include,excluded_reason=raw_qc or ('blank_control' if text(r['Media']).lower()=='blank' else ''),
            absolute_sum=sum(absolute.values()),contamination_status='not_assessable',modeling_action_default='review_required'))
        rel32={s:counts[s]/canonical for s in ALL} if canonical>0 else {}
        relative.append(meta(r)|dict(media=r['Media'],included=include)|rel32)
        if include:by_media[text(r['Media'])].append(rel32)
        assert np.isclose(sum(absolute.values()),final)
    means=[dict(Media=m,n_replicates=len(rr),**{s:float(np.mean([r[s] for r in rr])) for s in ALL}) for m,rr in by_media.items()]
    complete(filename,output,audit,[source],
        'Keep replicate-level format: relative reads over the same selected 40 species as original table; absolute proxy=raw Final OD times that within-replicate fraction. Retain negative endpoints and QC rows. Additional mean-relative table normalizes each replicate over canonical32, then averages equally, excluding blank, source Not_used/Contamination, any nonempty raw Note. At user request, source Not_used maps to the historical output label high other species; the raw exclusion reason is retained in the source workbook. This mapping restores the published label, not an independent contamination diagnosis.',fields=headers,
        extra={'growth_qc_counts':dict(Counter(r['growth_qc'] for r in output)),
               'consumer_note':'QC uses raw Note with the user-requested legacy mapping Not_used to high other species; Low_growth maps to low growth. No OD-derived QC is added.'})
