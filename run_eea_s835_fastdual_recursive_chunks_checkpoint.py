import argparse, json, time, gc
from pathlib import Path
from collections import Counter
from qiskit import QuantumCircuit
import eea_circuit_s835_fastdual as eea


def count_range(n:int,T_max:int,start:int,end:int,*,aux_size:int,measurement_uncompute:bool)->dict:
    cfg=eea.get_n_config(n); lw=int(cfg['len_width']); sw=int(cfg['shift_width'])
    eea.set_measurement_uncompute(measurement_uncompute)
    total=Counter(); num_qubits=None
    for T in range(start,end+1):
        qc=eea.build_step_circuit(n,T,T_max=T_max,aux_size=aux_size,measurement_uncompute=measurement_uncompute)
        total += eea.count_circuit_ops_recursive(qc)
        num_qubits=qc.num_qubits
        del qc
        if T%25==0: gc.collect()
    return {'mode':'eea-s835-fastdual-recursive-chunk','n':n,'T_max':T_max,'range':[start,end],
            'num_qubits':int(num_qubits or 0),'len_width':lw,'shift_width':sw,'aux_size':aux_size,
            'measurement_based':bool(measurement_uncompute),'ops':{k:int(v) for k,v in sorted(total.items())}}

def load_chunk(path:Path)->dict:
    return json.loads(path.read_text())

def write_sum(out_path:Path,*,n:int,T_max:int,chunks:list[dict],elapsed_s:float):
    total=Counter(); q=lw=sw=aux=None
    for c in chunks:
        total += Counter({str(k):int(v) for k,v in c['ops'].items()})
        q=int(c['num_qubits']); lw=int(c['len_width']); sw=int(c['shift_width']); aux=int(c['aux_size'])
    out={'mode':'eea-s835-fastdual-recursive-chunks-checkpointed','n':n,'T_max':T_max,
         'range':[1,T_max] if chunks and chunks[0]['range'][0]==1 and chunks[-1]['range'][1]==T_max else None,
         'num_qubits':q,'len_width':lw,'shift_width':sw,'aux_size':aux,
         'measurement_based':bool(chunks[0].get('measurement_based',False)) if chunks else None,
         'ops':{k:int(v) for k,v in sorted(total.items())},
         'chunks':[{'range':c['range'],'ops':{k:int(v) for k,v in c['ops'].items()}} for c in chunks],
         'elapsed_s_so_far':float(elapsed_s)}
    out_path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--n',type=int,default=256); ap.add_argument('--T-max',type=int,default=None)
    ap.add_argument('--chunk-size',type=int,default=200); ap.add_argument('--aux-size',type=int,default=None)
    ap.add_argument('--workdir',default='eea_s835_fastdual_chunks'); ap.add_argument('--out',default='eea_s835_fastdual_algorithm3_recursive_chunks_n256_measurement.json')
    ap.add_argument('--resume',action='store_true'); ap.add_argument('--measurement-uncompute',action='store_true')
    args=ap.parse_args(); cfg=eea.get_n_config(args.n); T_max=int(args.T_max or cfg['T_max']); lw=int(cfg['len_width']); sw=int(cfg['shift_width'])
    if args.aux_size is None: args.aux_size=int(eea.qiskit_paper_aux_size(args.n,lw,sw,T_max))
    workdir=Path(args.workdir); workdir.mkdir(parents=True,exist_ok=True); out_path=Path(args.out)
    chunks=[]; t0=time.perf_counter()
    for start in range(1,T_max+1,args.chunk_size):
        end=min(T_max,start+args.chunk_size-1); chunk_path=workdir/f'eea_s835_fastdual_n{args.n}_T{start:04d}_{end:04d}.json'
        if args.resume and chunk_path.exists() and chunk_path.stat().st_size>0:
            data=load_chunk(chunk_path); print(f"[resume] {start}-{end}: ccx={data['ops'].get('ccx',0)}",flush=True)
        else:
            print(f"[count] {start}-{end} aux={args.aux_size} measurement={args.measurement_uncompute}",flush=True)
            tic=time.perf_counter(); data=count_range(args.n,T_max,start,end,aux_size=args.aux_size,measurement_uncompute=args.measurement_uncompute)
            chunk_path.write_text(json.dumps(data,indent=2,sort_keys=True)+'\n')
            print(f"[done] {start}-{end}: {time.perf_counter()-tic:.2f}s ccx={data['ops'].get('ccx',0)}",flush=True)
        chunks.append(data); write_sum(out_path,n=args.n,T_max=T_max,chunks=chunks,elapsed_s=time.perf_counter()-t0)
    print(out_path)
if __name__=='__main__': main()
