"""Small analytic / independently computed checks for the SGIP port."""
from dataclasses import replace
import math
import numpy as np
import torch
from .physics import Physics, velocity, reconstruct, apply_reaction, resample, sample_field, ball
from .physics import reaction_value, reaction_derivative


def verify(device='cpu'):
    checks = {}
    x = torch.tensor([[0., 0., 0.], [.2, -.7, 1.3]], dtype=torch.float64, device=device)
    z = x.cpu().numpy()
    expected = np.column_stack((np.sin(z[:, 2]) + math.sqrt(1/3)*np.cos(z[:, 1]),
         math.sqrt(2/3)*np.sin(z[:, 0]) + np.cos(z[:, 2]),
         math.sqrt(1/3)*np.sin(z[:, 1]) + math.sqrt(2/3)*np.cos(z[:, 0])))
    np.testing.assert_allclose(velocity(x).cpu(), expected, atol=1e-14)
    checks['abc_components'] = True
    for kind in ('cubic', 'arrhenius'):
        c = Physics(kind, particles=20000, bins=8, half_width=8, initial_radius=2)
        values = torch.linspace(.001, .999, 201, dtype=torch.float64, device=device)
        h = 1e-6
        numerical = (reaction_value(values+h, kind) - reaction_value(values-h, kind))/(2*h)
        torch.testing.assert_close(numerical, reaction_derivative(values, kind), atol=2e-9, rtol=1e-6)
        updated = apply_reaction(values, c)
        residual = updated-values-c.dt*reaction_value(updated, kind)
        assert float(residual.abs().max()) < 1e-12
        checks[kind+'_implicit_residual'] = float(residual.abs().max())
        assert float(reaction_value(torch.tensor(0., device=device), kind)) == 0.
        points = torch.from_numpy(ball(c.particles, c.initial_radius, 123)).to(device)
        mass = 4/3*math.pi*c.initial_radius**3
        rho, ids, valid = reconstruct(points, mass, c)
        hist = np.histogramdd(points.cpu(), bins=[np.linspace(-8, 8, 9)]*3)[0]
        np.testing.assert_allclose(rho.cpu().numpy().reshape((8,)*3), hist*mass/c.particles/c.dx**3)
        assert abs(float(rho.sum())*c.dx**3-mass) < 1e-10
        updated = apply_reaction(rho, c)
        gen = torch.Generator(device=device).manual_seed(444)
        new = resample(points, updated, ids, valid, c, gen)
        assert new.shape == points.shape and torch.isfinite(new).all()
        new_hist = np.histogramdd(new.cpu(), bins=[np.linspace(-8, 8, 9)]*3)[0].ravel()/c.particles
        target = updated.cpu().numpy()/float(updated.sum())
        tv = .5*abs(new_hist-target).sum()
        assert tv < .035
        checks[kind+'_resampling_tv_20k'] = float(tv)
        sampled = sample_field(updated.cpu().numpy(), c, 1000, 55)
        assert np.isfinite(sampled).all() and abs(sampled).max() < c.half_width
    from .models import CompactCorrector
    model = CompactCorrector().to(device)
    torch.testing.assert_close(model(x.float(), torch.zeros((2,1), device=device)), x.float(), rtol=0, atol=0)
    checks['corrector_zero_head_identity'] = True
    return checks


if __name__ == '__main__':
    print(verify('cuda' if torch.cuda.is_available() else 'cpu'))
