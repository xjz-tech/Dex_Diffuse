from pathlib import Path
p=Path(__file__).resolve().parent
s=(p.parent/'server.py').read_text().replace("batch=len(history);seeds=[42+i for i in range(batch)]", "batch=len(history);seeds=msg.get('seeds',[42+i for i in range(batch)])")
s=s.replace("if j==0:controller.set_fixed_noise_from_seeds(seeds)","assert len(seeds)==batch")
s=s.replace("result,stats=controller.predict(history,target)","parts=[];befores=[];afters=[]\n                for lo in range(0,batch,9):\n                    controller.set_fixed_noise_from_seeds(seeds[lo:lo+9])\n                    part,stats=controller.predict(history[lo:lo+9],target[lo:lo+9]);parts.append(part);befores.append(stats.mse_before);afters.append(stats.mse_after)\n                result=np.concatenate(parts)")
s=s.replace("guidance_mse_before=stats.mse_before,guidance_mse_after=stats.mse_after", "guidance_mse_before=float(np.mean(befores)),guidance_mse_after=float(np.mean(afters))")
(p/'server.py').write_text(s)
