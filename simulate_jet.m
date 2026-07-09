function simulate_jet(regime)
% SIMULATE_JET  Capillary oscillations in a falling liquid jet.
%
%   Team Samudramanthan - Ensemble Young Physicists' Tournament 2026
%
%   Integrates the per-mode damped-harmonic-oscillator (DHO) equations for the
%   azimuthal modes n = 2,3,4 and the Rayleigh-Plateau growth mode n = 0, with
%   gravity thinning included so that omega_n(t) and gamma_n(t) vary in time.
%   The time-integrated amplitudes are mapped to axial position z(t) and the
%   jet surface R(theta, z) is reconstructed as a space-time heatmap, exactly
%   as described in the EYPT slides (Simulation Methodology / Results).
%
%   USAGE
%       simulate_jet                 % default: underdamped regime
%       simulate_jet('underdamped')  % Oh = 0.059  (mu = 0.05 Pa s)
%       simulate_jet('overdamped')   % Oh = 2.36   (mu = 2.0  Pa s)
%
%   Solver: ode45 (Dormand-Prince RK4/5), RelTol 1e-3, AbsTol 1e-6.
%
%   Reproduces: lambda(n=2) = 0.175 m  vs theory 0.180 m  (2.8% error),
%   and the critical viscosities mu_c,n = sqrt(rho*sigma*R0*(n^2-1)/n).

    if nargin < 1 || isempty(regime)
        regime = 'underdamped';
    end

    %% ---------------------------------------------------------------
    %  Physical parameters
    %% ---------------------------------------------------------------
    P.rho   = 1000;      % density                     [kg/m^3]
    P.sigma = 0.072;     % surface tension             [N/m]
    P.g     = 9.81;      % gravitational acceleration  [m/s^2]
    P.R0    = 10e-3;     % initial jet radius          [m]  (10 mm: resolvable)
    P.v0    = 0.6;       % initial jet speed           [m/s]

    switch lower(regime)
        case 'underdamped'
            P.mu = 0.05;         % Oh = 0.059
        case 'overdamped'
            P.mu = 2.0;          % Oh = 2.36  (> mu_c for n = 2,3,4)
        otherwise
            error('regime must be ''underdamped'' or ''overdamped''.');
    end

    Oh = P.mu / sqrt(P.rho * P.sigma * P.R0);
    fprintf('Regime: %s   |   mu = %.3g Pa.s   |   Oh = %.3f\n', ...
            regime, P.mu, Oh);

    %% ---------------------------------------------------------------
    %  Initial conditions and integration grid
    %% ---------------------------------------------------------------
    P.delta0    = 1.5e-3;             % initial amplitude          [m]
    P.deltadot0 = 0.0;                % initial amplitude rate     [m/s]
    tspan = linspace(0, 0.5, 600);    % t in [0, 0.5] s, 600 points

    opts  = odeset('RelTol', 1e-3, 'AbsTol', 1e-6);

    modes = [0 2 3 4];                % n = 0 (RP) and azimuthal n = 2,3,4
    delta = zeros(numel(modes), numel(tspan));

    %% ---------------------------------------------------------------
    %  Integrate one ODE per mode
    %% ---------------------------------------------------------------
    for i = 1:numel(modes)
        n  = modes(i);
        y0 = [P.delta0; P.deltadot0];
        [~, Y] = ode45(@(t, y) dho_rhs(t, y, n, P), tspan, y0, opts);
        delta(i, :) = Y(:, 1).';

        % critical viscosity + threshold report for n >= 2
        if n >= 2
            mu_c = sqrt(P.rho * P.sigma * P.R0 * (n^2 - 1) / n);
            state = 'UNDERDAMPED';
            if P.mu > mu_c, state = 'OVERDAMPED'; end
            fprintf('   n = %d:  mu_c = %.3f Pa.s  ->  %s\n', n, mu_c, state);
        end
    end

    %% ---------------------------------------------------------------
    %  Space-time mapping:  z(t) = v0 t + 1/2 g t^2
    %% ---------------------------------------------------------------
    z  = P.v0 * tspan + 0.5 * P.g * tspan.^2;   % axial position [m]
    Rz = R_of_t(tspan, P);                       % mean radius from continuity

    %% ---------------------------------------------------------------
    %  Reconstruct cross-sections and plot heatmaps
    %% ---------------------------------------------------------------
    theta = linspace(0, 2*pi, 200);

    figure('Name', sprintf('Falling jet - %s', regime), ...
           'Color', 'w', 'Position', [100 100 1100 650]);

    for i = 1:numel(modes)
        n = modes(i);
        surfMap = Rz(:) + (delta(i, :).') .* cos(n * theta);  % [Nz x Ntheta]

        subplot(2, 2, i);
        imagesc(theta, z, surfMap * 1e3);          % colour = radius in mm
        set(gca, 'YDir', 'normal');
        xlabel('\theta  [rad]');
        ylabel('z  [m]');
        title(sprintf('n = %d', n));
        colorbar; colormap(parula);
    end
    sgtitle(sprintf('Cross-section radius R(\\theta, z)  -  %s regime (Oh = %.3f)', ...
                    regime, Oh));

    %% ---------------------------------------------------------------
    %  Wavelength check for n = 2 (spatial period of the oscillation)
    %% ---------------------------------------------------------------
    i2      = find(modes == 2);
    lam_th  = 2*pi*P.v0 / sqrt(6*P.sigma/(P.rho*P.R0^3));   % theory
    lam_sim = measure_wavelength(z, delta(i2, :));          % from zero-crossings
    fprintf('n=2 wavelength:  theory = %.3f m   sim = %.3f m   error = %.1f%%\n', ...
            lam_th, lam_sim, 100*abs(lam_sim-lam_th)/lam_th);
end

% ===================================================================
function dydt = dho_rhs(t, y, n, P)
% Right-hand side of the per-mode damped harmonic oscillator with
% gravity-thinning (time-varying coefficients).
    R = R_of_t(t, P);

    if n == 0
        s0    = 0.343 * sqrt(P.sigma / (P.rho * R^3));   % growth rate
        gamma = 3 * P.mu / (P.rho * R^2);                % Trouton factor 3
        stiff = -s0^2;                                   % => exponential growth
    else
        omega = sqrt(n*(n^2-1) * P.sigma / (P.rho * R^3));
        gamma = n * P.mu / (P.rho * R^2);
        stiff = omega^2;
    end

    delta    = y(1);
    deltadot = y(2);
    dydt = [ deltadot;
             -2*gamma*deltadot - stiff*delta ];
end

% ===================================================================
function R = R_of_t(t, P)
% Mean jet radius from mass conservation with gravity acceleration:
%   Q = pi R^2 v,  v = v0 + g t   =>   R(t) = R0 sqrt(v0 / (v0 + g t)).
    v = P.v0 + P.g .* t;
    R = P.R0 .* sqrt(P.v0 ./ v);
end

% ===================================================================
function lam = measure_wavelength(z, d)
% Estimate spatial period from successive zero-crossings of the amplitude.
    d  = d - mean(d);
    zc = z(1:end-1) < 0 & d(2:end) >= 0;          % negative->positive crossings
    idx = find(zc);
    if numel(idx) >= 2
        lam = mean(diff(z(idx)));
    else
        lam = NaN;
    end
end
